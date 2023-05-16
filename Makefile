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
		-v `pwd`/models_weights:/sync_dir/models_weights \
		-v `pwd`/runs:/sync_dir/runs \
		amazon/aws-cli:2.7.19 \
		--endpoint-url https://eu-central-1.linodeobjects.com \
		s3 sync "s3://savant-data/articles/pruning/" /sync_dir/ --no-sign-request