from typing import Dict, List
import acl
import numpy as np
from threading import Lock

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

class ACLEngine(object):
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.load_input_dataset, self.load_output_dataset = None, None
        self.input_data:List[Dict[str,]] = []
        self.output_data:List[Dict[str,]] =  []
        self.lock = Lock()
        
        print('Start init resource')
        self.initResource()
        self.loadModel(model_path)
        self.allocateMem()
        print('Init resource success')

    def inference(self,frames:np.ndarray) -> List[np.ndarray]:
        bytes_data = frames.tobytes()
        np_ptr = acl.util.bytes_to_ptr(bytes_data)
        self.lock.acquire()
        ret = acl.rt.memcpy(self.input_data[0]["buffer"], self.input_data[0]["size"], np_ptr,self.input_data[0]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
        print('Start Inference')
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset,self.load_output_dataset)
        print(f'End Inference ret')
        inference_result = []
        for out in self.output_data:
            ret = acl.rt.memcpy(out['buffer_host'], out["size"],out["buffer"],out["size"],ACL_MEMCPY_DEVICE_TO_HOST)
            bytes_out = acl.util.ptr_to_bytes(out['buffer_host'], out["size"])
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data)
        self.lock.release()
        return inference_result

    def releaseResource(self):
        print('Start Resource destroyed')
        self.unloadModel()
        self.freeMem()
        self.destroyResource()
        print('Resource destroyed successfully')

    def initResource(self):
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        self.context,ret = acl.rt.create_context(self.device_id)
        self.stream, ret = acl.rt.create_stream()

    def loadModel(self, model_path):
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)

    def allocateMem(self):
        self.load_input_dataset = acl.mdl.create_dataset()
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.input_data = []

        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            self.input_data.append({"buffer": buffer, "size": buffer_size})

        self.load_output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.output_data = []
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            buffer_host, ret = acl.rt.malloc_host(buffer_size)
            self.output_data.append({"buffer": buffer, "size": buffer_size,'buffer_host':buffer_host})

    def unloadModel(self):
        ret = acl.mdl.unload(self.model_id)
        if self.model_desc:
            ret  = acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            self.context = None
        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)

    def freeMem(self):
        for item in self.input_data:
            ret = acl.rt.free(item["buffer"])
        ret = acl.mdl.destroy_dataset(self.load_input_dataset)
        for item in self.output_data:
            ret = acl.rt.free(item["buffer"])
            ret = acl.rt.free_host(item["buffer_host"])
        ret = acl.mdl.destroy_dataset(self.load_output_dataset)

    def destroyResource(self):
        ret = acl.rt.reset_device(self.device_id)
        ret = acl.finalize()
