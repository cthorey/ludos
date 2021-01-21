"""
Server to guess what ... serving ludos model.

This service is running a zmq client/server interface
to run the inference.

To access it, just connect to the server using

```
socket = context.socket(zmq.REQ)
socket.connect("tcp://IP_OF_THE_SERVER:PORT_OF_THE_SERVER")
```

The server expected request format and serialized in a specifc way.

The request should be a dict with three keys
1. model_id which reference the model to use
2. predict_method which store the name of the method you want to run
3. kwargs which store the argument of the method

Then this request should be pickled/compressed using

```
req = pickle.dumps(request, protocol)
req = zlib.compress(req)
```

before being sent to the

"""
import hashlib
import json
import logging
import pickle
import traceback
import zlib

import box
from box import Box

import zmq
from ludos.models import common


def get_logger():
    log = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    return log


class RequestFormatError(ValueError):
    pass


CODES = {
    'success': 200,
    'exceptions': {
        common.ModelLoadingError: 401,
        ValueError: 404,
        AttributeError: 404,
        TypeError: 404,
        RuntimeError: 404,
        box.exceptions.BoxKeyError: 401,
        RequestFormatError: 402
    }
}


class LudosServer(object):
    """
    Simple server exposing models inference via a client/server zeroMQ interface.

    The server expected request format and serialized in a specifc way.

    The request should be a dict with three keys
    1. model_id: Name of the model in the registry
    2. predict_method: which store the name of the method you want to run
    3. predict_kwargs: which store the argument of the method

    Then this request should be pickled/compressed before being sent to the
    server. Inversely, the response should also be decrompress/unserialized
    using pickle

    Full client side workflow below:

    ```
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://IP_OF_THE_SERVER:PORT_OF_THE_SERVER")
    req = zlib.compress(pickle.dumps(request, protocol))
    socket.send(req,flags = 0)
    response = socket.recv(flags = 0)
    response = pickle.loads(zlib.decompress(response))
    ```

    """
    def __init__(self, host: str = '*', port: int = 5555):
        """
        Args:
            host (str): IP address of the host
            port (int): Port to access the server
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://{}:{}".format(host, port))
        self.models = Box()
        self.host = host
        self.port = port
        self.logger = get_logger()

    def send_response(self,
                      payload,
                      status_code: int,
                      protocol: int = 2,
                      flags: int = 0):
        """
        Send the response to the client

        Args:
            payload (obj): output of the model
            status_code (int): exit status
            protocol (int): Protocol to pickle the msg. Use 2
            to talk to python2
        """
        response = dict()
        response['payload'] = payload
        response['status_code'] = status_code
        p = pickle.dumps(response, protocol)
        z = zlib.compress(p)
        return self.socket.send(z, flags=flags)

    def receive_request(self, flags: int = 0):
        """
        Receive request.

        A request is made of three attributes
        request = dict(model_id='cctv_expert',
                       predict_method='predict_prob',
                       predict_kwargs=...)

        Returns:
            Deserialized request sent by the client.
        """
        z = self.socket.recv(flags)
        p = zlib.decompress(z)
        request = pickle.loads(p, encoding='latin1')
        for key in ['model_id', 'predict_method', 'predict_kwargs']:
            if key not in request.keys():
                self.logger.error('Missing key {}'.format(key))
                raise RequestFormatError()
        return request

    def load_model(self, model_id: str, model_task: str, model_name: str,
                   expname: str):
        """
        Args:
            model_id (str): model key
            model_task (str): Task for the model
            model_name (str): Name of the model
            expname (str): Name of the experiment

        Returns:
            model
        """
        self.logger.info('Loading model {}/{}/{}'.format(
            model_task, model_name, expname))
        self.models[model_id] = common.load_model(model_task=model_task,
                                                  model_name=model_name,
                                                  expname=expname)
        self.logger.info('Succesfully load model {}/{}/{}'.format(
            model_task, model_name, expname))

    def start(self):
        """
        Start the server loop
        """
        self.logger.info('Server started on http://{}:{}'.format(
            self.host, self.port))
        while True:
            try:
                self.logger.info('Waiting new request')
                request = self.receive_request()
                self.logger.info('Running inference')
                out = getattr(
                    self.models[request['model_id']],
                    request['predict_method'])(**request['predict_kwargs'])
            except (common.ModelLoadingError, RequestFormatError, ValueError,
                    TypeError, RuntimeError, AttributeError,
                    box.exceptions.BoxKeyError) as e:
                trace = traceback.format_exc()
                code_status = CODES['exceptions'][e.__class__]
                self.logger.error('Error with status: {}'.format(code_status))
                self.logger.error('Traceback: {}'.format(trace))
                self.send_response(payload='', status_code=code_status)
                continue
            except Exception as e:
                trace = traceback.format_exc()
                self.logger.error('Traceback: {}'.format(trace))
                code_status = 404
                self.send_response(payload='', status_code=code_status)
                continue
            self.send_response(payload=out, status_code=CODES['success'])
            self.logger.info('Succesfully run inference')
