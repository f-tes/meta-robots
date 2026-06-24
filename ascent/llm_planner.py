from typing import Dict, Tuple, Any, Union, List
import numpy as np
import cv2
import os
import logging
from model_api.qwen25_out import Qwen2_5Client
from ascent.mapping.object_point_cloud_map import ObjectPointCloudMap
from ascent.mapping.obstacle_map import ObstacleMap
from ascent.mapping.value_map import ValueMap
import json
from skimage.metrics import structural_similarity as ssim
from constants import (
    INDENT_L1,
    INDENT_L2,
    DIRECT_MAPPING,
    REFERENCE_ROOMS,
    STICKY_FRONTIER_DISTANCE_THRESHOLD,
    STICKY_FRONTIER_STEP_THRESHOLD,
    REPEATED_SELECTION_THRESHOLD,
    MULTI_FLOOR_ASK_STEP_THRESHOLD,
    FLOOR_EXP_STEP_THRESHOLD,
)
import networkx as nx
from ascent.harness_bridge import get_harness

class Ascent_LLM_Planner:
    def __init__(self, num_envs=1, nearby_distance = 3.0, topk = 3, target_object_list = [""], floor_probabilities_df=None):

        self._num_envs = num_envs
        self._force_frontier = [np.zeros(2) for _ in range(self._num_envs)]
        self.nearby_distance = nearby_distance
        self.topk = topk
        self.frontier_step_list = [[] for _ in range(self._num_envs)]
        self.vlm_response = ["" for _ in range(self._num_envs)]
        self._last_value = [float("-inf") for _ in range(self._num_envs)]
        self._last_frontier = [np.zeros(2) for _ in range(self._num_envs)]
        self._target_object = target_object_list
        self._llm = Qwen2_5Client(port=int(os.environ.get("QWEN2_5_PORT", "13181")))
        self.multi_floor_ask_step = [0 for _ in range(self._num_envs)]
        self.floor_probabilities_df = floor_probabilities_df
        self.frontier_rgb_list = [[] for _ in range(self._num_envs)]
        ## knowledge graph
        with open('statistic_priors/knowledge_graph.json', 'r') as f:
            self.knowledge_graph = nx.node_link_graph(json.load(f))
        self.floor_num = [1 for _ in range(self._num_envs)]
    def reset(self, env):
        # 防止来回走动
        self._force_frontier[env] = np.zeros(2)
        self.frontier_step_list[env] = []        
        self.vlm_response[env] = ""
        self._last_value[env] = float("-inf")
        self._last_frontier[env] = np.zeros(2)
        self._target_object[env] = ""
        self.multi_floor_ask_step[env] = 0
        self.frontier_rgb_list[env] = []
        self.floor_num[env] = 1

    def _get_best_frontier_with_llm(
            self,
            observations_cache: List[dict], 
            obstacle_map: List[ObstacleMap],
            value_map: List[ValueMap],
            object_map: List[ObjectPointCloudMap],
            obstacle_map_list: List[List[ObstacleMap]],
            value_map_list: List[List[ValueMap]],
            object_map_list: List[List[ObjectPointCloudMap]],
            frontiers: np.ndarray,
            env: int = 0,
            topk: int = 3,
            use_multi_floor: bool = True,
            floor_num: List[int] = [1],
            cur_floor_index: List[int] = [],
            num_steps: List[int] = [1],
            last_frontier_distance: List[float] = [1],
            frontier_stick_step: List[int] = [1],
        ) -> Tuple[np.ndarray, float]:
            """Returns the best frontier and its value based on self._value_map.

            Returns:
                Tuple[np.ndarray, float]: The best frontier and its value.
            """
            # 🆕 0. 如果只有一个前沿点，直接导航到该点
            if len(frontiers) == 1:
                return frontiers[0], 1.0
            
            # 1. 初始化
            sorted_pts, sorted_values = self._sort_frontiers_by_value(obstacle_map, value_map, frontiers, env)
            robot_xy = observations_cache[env]["robot_xy"]

            # DP1: compute_frontier_value — apply distance-based bonus and re-sort
            if len(sorted_pts) > 0:
                harness = get_harness()
                dists_for_dp1 = [float(np.linalg.norm(pt - robot_xy)) for pt in sorted_pts]
                enhanced = [
                    harness.compute_frontier_value(float(val), d)
                    for val, d in zip(sorted_values, dists_for_dp1)
                ]
                order = np.argsort([-v for v in enhanced])
                print(f"[DP1] frontier scores (raw→enhanced, dist): " +
                      ", ".join(f"{sorted_values[i]:.3f}→{enhanced[i]:.3f}@{dists_for_dp1[i]:.1f}m"
                                for i in range(len(sorted_pts))))
                sorted_pts = sorted_pts[order]
                sorted_values = [enhanced[i] for i in order]
                try:
                    from ascent.harness_bridge import safe_emit
                    safe_emit("on_frontier_evaluated", sorted_pts.tolist(), sorted_values, env)
                except Exception:
                    pass

            best_frontier, best_value = None, None

            # 2. 处理强制前沿
            best_frontier, best_value = self._try_force_frontier(sorted_pts, sorted_values, env)
            if best_frontier is not None:
                print(f"Force Move.")

            # 3. 处理近邻前沿 (如果未选中强制前沿且满足条件)
            # 只有在没有强制前沿，并且第一次探索完成后才尝试近邻
            if best_frontier is None and obstacle_map[env]._finish_first_explore:
                best_frontier_nearby, best_value_nearby, activated_neighbor_search = self._try_nearby_frontier(sorted_pts, sorted_values, robot_xy, env)

                if activated_neighbor_search:
                    obstacle_map[env]._neighbor_search = True
                    best_frontier, best_value = best_frontier_nearby, best_value_nearby
                    print(f"Frontier {best_frontier} is very close (distance: {np.linalg.norm(best_frontier - robot_xy):.2f}m), selecting it.")
                else:
                    # 如果尝试近邻搜索但没有找到合适的近邻前沿，则将 _finish_first_explore 设为 False
                    # 这会促使系统在下一个周期重新评估并强制一个探索点
                    obstacle_map[env]._finish_first_explore = False
                    obstacle_map[env]._neighbor_search = False # 确保如果未激活近邻搜索，该标志为 False

            # 4. LLM 决策 (如果前沿仍未选中)
            if best_frontier is None:
                best_frontier, best_value = self._decide_frontier_with_llm(obstacle_map, object_map, sorted_pts, sorted_values, env, topk, use_multi_floor,
                                                                           floor_num, cur_floor_index, num_steps,obstacle_map_list,object_map_list,robot_xy)

            # 5. 处理前沿点粘滞/循环检测和禁用
            # 这一部分逻辑相对独立且复杂，可以封装
            self._handle_frontier_stick_and_disable(best_frontier, robot_xy, env, last_frontier_distance, frontier_stick_step, obstacle_map)

            # 6. 更新状态并返回
            self._last_value[env] = best_value
            self._last_frontier[env] = best_frontier
            
            if not obstacle_map[env]._finish_first_explore:
                obstacle_map[env]._finish_first_explore = True
                self._force_frontier[env] = best_frontier.copy()
                
            print(f"Now the best_frontier is {best_frontier}")
            return best_frontier, best_value
    
    def _sort_frontiers_by_value(
        self, obstacle_map, value_map, frontiers: np.ndarray, env: int = 0,
    ) -> Tuple[np.ndarray, List[float]]:

        # 我们可以将它们打包成 (point, value) 对进行过滤        
        # 步骤1: 获取初始排序后的前沿点和值
        raw_sorted_pts, raw_sorted_values = value_map[env].sort_waypoints(frontiers, 0.5)
        # 步骤2: 过滤掉禁用的前沿点，并同时保留对应的值
        filtered_pairs = []
        for pt, val in zip(raw_sorted_pts, raw_sorted_values):
            if tuple(pt) not in obstacle_map[env]._disabled_frontiers:
                filtered_pairs.append((pt, val))
        # 步骤3: 解包过滤后的结果
        if not filtered_pairs: # 如果所有前沿都被禁用，返回空列表
            return np.array([]), []
        sorted_frontiers = np.array([pair[0] for pair in filtered_pairs])
        sorted_values = [pair[1] for pair in filtered_pairs]        
        return sorted_frontiers, sorted_values
    
    @staticmethod
    def hamming_distance(hash1, hash2):
        """
        计算两个pHash值的汉明距离。
        """
        return np.sum(hash1 != hash2)

    def _try_force_frontier(self, sorted_pts, sorted_values, env):
        # 提取强制前沿的逻辑
        force_frontier = self._force_frontier[env]
        if np.any(force_frontier): # 检查是否是有效的强制前沿
            for i, frontier in enumerate(sorted_pts):
                if np.array_equal(frontier, force_frontier):
                    return frontier, sorted_values[i]
        return None, None

    def _try_nearby_frontier(self, sorted_pts, sorted_values, robot_xy, env):
        # 提取近邻前沿的逻辑
        distances = [np.linalg.norm(frontier - robot_xy) for frontier in sorted_pts]
        close_frontiers_info = [
            (idx, frontier, distance)
            for idx, (frontier, distance) in enumerate(zip(sorted_pts, distances))
            if distance <= self.nearby_distance
        ]

        if close_frontiers_info:
            closest_frontier_info = min(close_frontiers_info, key=lambda x: x[2])
            best_frontier_idx = closest_frontier_info[0]
            return sorted_pts[best_frontier_idx], sorted_values[best_frontier_idx], True # 返回是否激活了近邻搜索
        return None, None, False

    def _decide_frontier_with_llm(self, obstacle_map, object_map, sorted_pts, sorted_values, env, topk, use_multi_floor, floor_num, cur_floor_index, num_steps, obstacle_map_list, object_map_list, robot_xy=None):
        # 提取 LLM 决策逻辑，包括多楼层和单楼层
        if len(sorted_pts) == 0: # 没有可选前沿，直接返回 None
            return None, 0.0 # 或者一个默认的低值

        # 单个前沿点的情况
        if len(sorted_pts) == 1:
            self._last_value[env] = sorted_values[0]
            self._last_frontier[env] = sorted_pts[0]
            return sorted_pts[0], sorted_values[0]

        # 多个前沿点，准备 LLM 输入
        self.frontier_step_list[env] = []
        frontier_index_list = []

        # DP4: filter_diverse_frontiers — gather candidates then deduplicate
        n_candidates = min(topk * 3, len(sorted_pts))
        raw_candidates = []
        for idx in range(n_candidates):
            floor_num_steps, image_rgb = obstacle_map[env].extract_frontiers_with_image(sorted_pts[idx])
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            raw_candidates.append((idx, image_gray, floor_num_steps))

        selected = get_harness().filter_diverse_frontiers(raw_candidates, topk)
        for rank_idx, step in selected:
            self.frontier_step_list[env].append(step)
            frontier_index_list.append(rank_idx)

        target_object_category = self._target_object[env].split("|")[0]

        best_frontier_idx = 0 # 默认值

        # DP2: should_trigger_llm — optional additional gate
        distances = [float(np.linalg.norm(sorted_pts[i] - robot_xy)) for i in frontier_index_list] if (frontier_index_list and robot_xy is not None) else []
        _vals_for_dp2 = [sorted_values[i] for i in frontier_index_list]
        _dp2_fire = get_harness().should_trigger_llm(_vals_for_dp2, distances, len(frontier_index_list))
        print(f"[DP2] should_trigger_llm={_dp2_fire}  values={[f'{v:.3f}' for v in _vals_for_dp2]}  dists={[f'{d:.1f}m' for d in distances]}")
        if not _dp2_fire:
            # Skip LLM; use top-value frontier
            best_frontier_idx = frontier_index_list[0] if frontier_index_list else 0
        # DP3: should_trigger_multifloor_llm — gates inter-floor LLM call
        elif get_harness().should_trigger_multifloor_llm(
            floor_num[env],
            num_steps[env] - self.multi_floor_ask_step[env],
            obstacle_map[env]._floor_num_steps,
            use_multi_floor,
        ):
            self.multi_floor_ask_step[env] = num_steps[env]
            # DP6: build_interfloor_prompt
            current_floor = cur_floor_index[env] + 1
            total_floors = self.floor_num[env]
            floor_probs = self.get_floor_probabilities(self.floor_probabilities_df, target_object_category, total_floors)
            room_probs = self.get_room_probabilities(target_object_category)
            floor_descs = []
            for fl in range(total_floors):
                try:
                    rooms = object_map_list[env][fl].this_floor_rooms or {"unknown rooms"}
                    objects = object_map_list[env][fl].this_floor_objects or {"unknown objects"}
                    floor_descs.append({
                        "floor_id": fl + 1,
                        "status": "Current floor" if fl + 1 == current_floor else "Other floor",
                        "fully_explored": obstacle_map_list[env][fl]._this_floor_explored,
                        "room": ", ".join(rooms),
                        "objects": ", ".join(objects),
                    })
                except Exception as e:
                    logging.error(f"Error describing floor {fl}: {e}")
            multi_floor_prompt = get_harness().build_interfloor_prompt(
                target_object_category, current_floor, total_floors,
                floor_probs, room_probs, floor_descs,
            )
            print(f"## Multi-floor Prompt:\n{multi_floor_prompt}")
            multi_floor_response = self._llm.chat(multi_floor_prompt)
            print(f"## Multi-floor Raw Response:\n{multi_floor_response}")
            try:
                from ascent.harness_bridge import safe_emit
                safe_emit("on_llm_call", multi_floor_prompt, multi_floor_response, "interfloor", env)
            except Exception:
                pass

            if multi_floor_response == "-1":
                print("[DP8] parse_interfloor_response: LLM timeout, falling back to single-floor")
                best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list, obstacle_map, object_map)
            else:
                # DP8: parse_interfloor_response
                target_floor, reason = get_harness().parse_interfloor_response(
                    multi_floor_response, current_floor, total_floors
                )
                print(f"[DP8] parse_interfloor_response: target_floor={target_floor} current={current_floor} reason={reason!r}")
                if reason:
                    response_str = f"Floor Index: {target_floor}. Reason: {reason}"
                    self.vlm_response[env] = "## Multi-floor Prompt:\n" + response_str
                    print(f"## Multi-floor Response:\n{response_str}")
                if target_floor > current_floor:
                    return sorted_pts[0], -100  # 上楼
                elif target_floor < current_floor:
                    return sorted_pts[0], -200  # 下楼
                else:
                    best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list, obstacle_map, object_map)
        else: # 单楼层决策
            best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list, obstacle_map, object_map)

        return sorted_pts[best_frontier_idx], sorted_values[best_frontier_idx]

    def _handle_frontier_stick_and_disable(self, current_best_frontier, robot_xy, env, last_frontier_distance, frontier_stick_step, obstacle_map,):
        # 将前沿点粘滞和禁用逻辑封装
        if np.array_equal(self._last_frontier[env], current_best_frontier):
            if frontier_stick_step[env] == 0:
                last_frontier_distance[env] = np.linalg.norm(current_best_frontier - robot_xy)
                frontier_stick_step[env] += 1
            else:
                current_distance = np.linalg.norm(current_best_frontier - robot_xy)
                print(f"Distance Change: {np.abs(last_frontier_distance[env] - current_distance)} and Stick Step {frontier_stick_step[env]}")
                if np.abs(last_frontier_distance[env] - current_distance) > STICKY_FRONTIER_DISTANCE_THRESHOLD and not obstacle_map[env]._neighbor_search:
                    frontier_stick_step[env] = 0
                    last_frontier_distance[env] = current_distance
                else:
                    if frontier_stick_step[env] >= STICKY_FRONTIER_STEP_THRESHOLD:
                        obstacle_map[env]._disabled_frontiers.add(tuple(current_best_frontier))
                        print(f"Frontier {current_best_frontier} is disabled due to no movement.")
                        frontier_stick_step[env] = 0
                    else:
                        frontier_stick_step[env] += 1
        else:
            frontier_stick_step[env] = 0
            last_frontier_distance[env] = 0
            if tuple(current_best_frontier) in obstacle_map[env]._best_frontier_selection_count:
                self._force_frontier[env] = current_best_frontier.copy()

        # 选中次数统计和禁用
        frontier_tuple = tuple(current_best_frontier)
        obstacle_map[env]._best_frontier_selection_count.setdefault(frontier_tuple, 0)
        if not np.array_equal(self._last_frontier[env], current_best_frontier): # 只有非连续选中才增加计数
            obstacle_map[env]._best_frontier_selection_count[frontier_tuple] += 1
            if obstacle_map[env]._best_frontier_selection_count[frontier_tuple] >= REPEATED_SELECTION_THRESHOLD:
                obstacle_map[env]._disabled_frontiers.add(frontier_tuple)
                print(f"Frontier {current_best_frontier} is disabled due to repeated non-consecutive selection.")

    def llm_analyze_single_floor(self, env, target_object_category, frontier_index_list, obstacle_map, object_map):
        """
        Analyze the environment using the Large Language Model (LLM) to determine the best frontier to explore.

        Parameters:
        env (str): The current environment identifier.
        target_object_category (str): The category of the target object to find.
        frontier_identifiers (list): A list of frontier identifiers (e.g., ["A", "B", "C", "P"]).
        exploration_status (str): A binary string representing the exploration status of each floor.

        Returns:
        str: The identifier of the frontier that is most likely to lead to the target object.
        """
    
        # else, continue to explore on this floor
        prompt = self._prepare_single_floor_prompt(target_object_category, env, obstacle_map, object_map)

        # Analyze the environment using the VLM
        print(f"## Single-floor Prompt:\n{prompt}")
        response = self._llm.chat(prompt)
        print(f"## Single-floor Raw Response:\n{response}")
        try:
            from ascent.harness_bridge import safe_emit
            safe_emit("on_llm_call", prompt, response, "intrafloor", env)
        except Exception:
            pass

        # DP7: parse_intrafloor_response
        if response == "-1":
            print("[DP7] parse_intrafloor_response: LLM timeout, defaulting to frontier 0")
            temp_frontier_index = 0
        else:
            temp_frontier_index, reason = get_harness().parse_intrafloor_response(
                response, len(frontier_index_list)
            )
            if reason:
                response_str = f"Area Index: {temp_frontier_index + 1}. Reason: {reason}"
                self.vlm_response[env] = "## Single-floor Prompt:\n" + response_str
                print(f"## Single-floor Response:\n{response_str}")
            print(f"[DP7] parse_intrafloor_response: index={temp_frontier_index} reason={reason!r}")
        
        return frontier_index_list[temp_frontier_index]

    def get_room_probabilities(self, target_object_category: str):
        """
        获取目标对象类别在各个房间类型的概率。
        
        :param target_object_category: 目标对象类别
        :return: 房间类型概率字典
        """
        # 定义一个映射表，用于扩展某些目标对象类别的查询范围
        synonym_mapping = {
            "couch": ["sofa"],
            "sofa": ["couch"],
            # 可以根据需要添加更多映射关系
        }

        # 获取目标对象类别及其同义词
        target_categories = [target_object_category] + synonym_mapping.get(target_object_category, [])

        # 如果目标对象类别及其同义词都不在知识图谱中，直接返回空字典
        if not any(category in self.knowledge_graph for category in target_categories):
            return {}

        room_probabilities = {}
        for room in REFERENCE_ROOMS:
            for category in target_categories:
                if self.knowledge_graph.has_edge(category, room):
                    room_probabilities[room] = round(self.knowledge_graph[category][room]['weight'] * 100, 1)
                    break  # 找到一个有效类别后，不再检查其他类别
            else:
                room_probabilities[room] = 0.0
        return room_probabilities

    def get_floor_probabilities(self, df, target_object_category, total_floor):
        """
        获取当前楼层和场景的物体分布概率。

        Parameters:
        df (pd.DataFrame): 包含物体分布概率的表格。
        target_object_category (str): 目标物体类别。
        total_floor (int): 场景中的总楼层数。

        Returns:
        dict: 所有相关楼层的物体分布概率，包含缺失楼层的0.0值。
        """
        if df is None or target_object_category not in df.index:
            return {i: 0.0 for i in range(1, total_floor + 1)}
        
        probabilities = {}
        max_possible_floor = 0
        
        # 首先确定表格中存在的最大楼层数
        for col in df.columns:
            if col.startswith("train_floor"):
                current_floor = int(col.split('_')[1])
                if current_floor > max_possible_floor:
                    max_possible_floor = current_floor
        
        # 处理两种情况：请求楼层数超过表格最大支持或正常情况
        if total_floor > max_possible_floor:
            # 只处理表格中存在的楼层
            for y in range(1, max_possible_floor + 1):
                col_name = f"train_floor{max_possible_floor}_{y}"
                probabilities[y] = df.set_index("category").at[target_object_category, col_name] if col_name in df else 0.0
        else:
            # 正常处理请求的楼层数
            for y in range(1, total_floor + 1):
                col_name = f"train_floor{total_floor}_{y}"
                probabilities[y] = df.set_index("category").at[target_object_category, col_name] if col_name in df else 0.0
        
        return probabilities
    
    def _prepare_single_floor_prompt(self, target_object_category, env, obstacle_map, object_map):
        """Prepare prompt for single-floor frontier selection. Delegates to DP5."""
        area_descriptions = []
        self.frontier_rgb_list[env] = []
        for i, step in enumerate(self.frontier_step_list[env]):
            try:
                room = object_map[env].each_step_rooms[step] or "unknown room"
                objects = object_map[env].each_step_objects[step] or "no visible objects"
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                self.frontier_rgb_list[env].append(obstacle_map[env]._each_step_rgb[step])
                area_descriptions.append({"area_id": i + 1, "room": room, "objects": objects})
            except (IndexError, KeyError) as e:
                logging.warning(f"Error accessing room or objects for step {step}: {e}")
                continue
        room_probabilities = self.get_room_probabilities(target_object_category)
        # DP5: build_intrafloor_prompt
        return get_harness().build_intrafloor_prompt(target_object_category, area_descriptions, room_probabilities)
    
    def _prepare_multiple_floor_prompt(self, target_object_category, env, cur_floor_index, obstacle_map_list, object_map_list):
        """Multi-floor prompt builder — now routes through DP6 (kept for any legacy callers)."""
        current_floor = cur_floor_index[env] + 1
        total_floors = self.floor_num[env]
        floor_probs = self.get_floor_probabilities(self.floor_probabilities_df, target_object_category, total_floors)
        room_probs = self.get_room_probabilities(target_object_category)
        floor_descs = []
        for fl in range(total_floors):
            try:
                rooms = object_map_list[env][fl].this_floor_rooms or {"unknown rooms"}
                objects = object_map_list[env][fl].this_floor_objects or {"unknown objects"}
                floor_descs.append({
                    "floor_id": fl + 1,
                    "status": "Current floor" if fl + 1 == current_floor else "Other floor",
                    "fully_explored": obstacle_map_list[env][fl]._this_floor_explored,
                    "room": ", ".join(rooms),
                    "objects": ", ".join(objects),
                })
            except Exception as e:
                logging.error(f"Error describing floor {fl}: {e}")
        return get_harness().build_interfloor_prompt(
            target_object_category, current_floor, total_floors, floor_probs, room_probs, floor_descs
        )

    def _format_probs(self, prob_dict):
        """概率格式化工具（复用单楼层风格）"""
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"- {k}: {v:.1f}%" for k, v in sorted_probs])

    def _extract_multiple_floor_decision(self, multi_floor_response, env, cur_floor_index) -> int:
        """Routes to DP8 (parse_interfloor_response) — kept for legacy callers."""
        current_floor = cur_floor_index[env] + 1
        target_floor, reason = get_harness().parse_interfloor_response(
            multi_floor_response, current_floor, self.floor_num[env]
        )
        if reason:
            response_str = f"Floor Index: {target_floor}. Reason: {reason}"
            self.vlm_response[env] = "## Multi-floor Prompt:\n" + response_str
            print(f"## Multi-floor Response:\n{response_str}")
        return target_floor

