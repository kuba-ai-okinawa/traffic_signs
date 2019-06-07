FROM aiokinawa/traffic_signs:001

LABEL name="aiokinawa/traffic_signs"
LABEL version="latest"

# Copy code and data
#COPY . /app
COPY ./environment_linux.yaml /app/

# Set working directory to project's dir
WORKDIR /app

# Update conda environment
RUN conda env update -f ./environment_linux.yaml

# Run the app
CMD  python ./scripts/run_server.py --config=./configurations/development_config.yaml
