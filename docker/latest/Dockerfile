FROM aiokinawa/traffic_signs:001

LABEL name="aiokinawa/traffic_signs"
LABEL version="latest"

# Copy code and data
COPY . /app_user/projects/code/traffic_signs
COPY $TRAFFIC_SIGNS_MODEL_PATH /app_user/projects/data/traffic-signs-model.h5

# Set working directory to project's dir
WORKDIR /app_user/projects/code/traffic_signs

# Update conda environment
RUN conda env update -f ./environment_linux.yaml
