import requests
requests.packages.urllib3.disable_warnings()
import requests.adapters
old_send = requests.adapters.HTTPAdapter.send
def new_send(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_send(self, *args, **kwargs)
requests.adapters.HTTPAdapter.send = new_send

import runpy
runpy.run_module('model_api.blip2itm_out', run_name='__main__')
