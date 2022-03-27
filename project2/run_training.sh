echo "Building container"
docker build -t rl .
docker run -v /Users/jorgenr/Code/School/IT3105/project2:/project2 rl