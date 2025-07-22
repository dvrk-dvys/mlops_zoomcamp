LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=stream-model-duration:${LOCAL_TAG}

test:
	cd module_6/src && python -m pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y . --disable=C0114,C0115,C0116,C0103,W0621,E0401,W0611,C0415,W0603,C0209,R0801,W1309,W0613,R0903,C0301,C0413,W0404,E0602,W0104,W0106,C0121,W1508,W0602,W0612,W0105,W3101,E0611,W1203,W0718,W0707,W0719,R0913,R0917,W1514,R1705,R1722,C0412,C0411

build: test quality_checks
	cd module_6/src && docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} cd module_6/integration-test && ./run.sh

publish: build integration_test
	echo "Publishing ${LOCAL_IMAGE_NAME}"
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} ./module_6/scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
