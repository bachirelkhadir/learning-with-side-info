build-docker-image:
	docker build  --build-arg USER_ID=$$(id -u) --build-arg GROUP_ID=$$(id -g) -t learningwithds  .

build-docker-image-no-cache:
	docker build --no-cache --build-arg USER_ID=$$(id -u) --build-arg GROUP_ID=$$(id -g) -t learningwithds  .

run-docker-image:
	docker run -p 8888:8888 -it --rm -v $(PWD)/../:/home/user/project learningwithds bash

jupyter:
	docker run -p 8888:8888 -it --rm -v $(PWD)/../:/home/user/project learningwithds julia -e 'using Pkg; Pkg.activate("env"); using IJulia; notebook(dir="/home/user/project");'
# end
