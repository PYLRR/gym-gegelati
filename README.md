# gym-wrapper
The purpose of this repository is to link the TPG [GEGELATI library](https://github.com/gegelati/gegelati) to the [gym environment](https://gym.openai.com).

gym is a python library. As a result, it needs a special binding to be used in C++. To do so, http binding is used : it needs a server from the python side, and a client (using curl) from the C++ side. The code of these two components has been taken from [gym-http-api](https://github.com/openai/gym-http-api), before being modified in accordance to the free software licence.

To finish, the http api uses another library to read the json http answers : [jsoncpp](https://github.com/open-source-parsers/jsoncpp).

## How to install ?
First of all, install the [GEGELATI library](https://github.com/gegelati/gegelati). 
To do so, execute the following commands :
```
$ git clone https://github.com/gegelati/gegelati.git
$ cd gegelati/bin
$ cmake ..
$ cmake --build . --target INSTALL # On Windows
$ cmake --build . --target install # On Linux
```

Then, clone this [project](https://github.com/PYLRR/gym-gegelati).
```
$ git clone https://github.com/PYLRR/gym-gegelati.git
```

The python part needs some dependencies. 
```
$ cd gym-gegelati/server-python
$ pip install -Ir requirements.txt 
```

## How to use ?
Start the python server:
```
$ cd gym-gegelati/server-python
$ python3 gym_http_server.py
```

Then, simply use the main of the C++ side. This will use a "Learning Agent" from Gegelati and make it work on Gym by communicating with the python server.

NB:
In case of parallel learning, to enable the server to handle enough requests is it advised to switch on the multithreading for the server, i.e. changing the startup of the server (python side) to :
```
app.run(host=args.listen, port=args.port, threaded=True)
```
However, this will not work well with rendering. In case we want to render a gym application, put threaded to False.

## License
This project is free. Any copy and modification is allowed with the only restriction it has to be in turn free (see LICENSE file).
