# traffic_signs

Data: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip

To train model:
`python ./scripts/train.py --config_path=/path/to/config`

To run web server:
`python ./scripts/run_server.py --config_path=/path/to/config`

### To build docker containers:

- Build base container that installs anaconda  
`docker build --tag aiokinawa/traffic_signs:001 -f ./docker/001/Dockerfile .`

- Build container that updates code, data and anaconda environment  
`docker build --tag aiokinawa/traffic_signs:latest -f ./docker/latest/Dockerfile .`

- Start container and run flask server:  
`docker run -it --rm -p 5000:5000 aiokinawa/traffic_signs:latest`

- Start container and log inside it:  
`docker run -it --rm -p 5000:5000 aiokinawa/traffic_signs:latest bash`
