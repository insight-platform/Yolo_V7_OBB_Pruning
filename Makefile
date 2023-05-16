docker-image-build-notebook:
	DOCKER_BUILDKIT=1 docker build --target yolov7obb -t yolov7obb:1.0 --build-arg UID=`id -u` --build-arg GID=`id -g` \
--build-arg TZ=`cat /etc/timezone` --progress=plain . && docker tag yolov7obb:1.0 yolov7obb:latest


## Up rules
notebook-up:
	docker-compose up -d yolov7obb

# Down rules
notebook-down:
	docker-compose rm -s -v yolov7obb

download-data:
	docker run --rm \
		-v `pwd`/mpdels_weights:/sync_dir/data \
		-v `pwd`/models:/sync_dir/models \
		-v `pwd`/modules:/sync_dir/modules \
		-v `pwd`/tests:/sync_dir/tests \
		-v `pwd`/model_storage.ini:/opt/app/model_storage.ini \
		-e AWS_CONFIG_FILE=/opt/app/model_storage.ini \
		amazon/aws-cli:2.7.19 \
		--endpoint-url `awk -F "=" '/endpoint_url/ {print $$2}' model_storage.ini` \
		s3 sync "s3://ml-data/motion-insights/framework-examples" /sync_dir/