echo "Building container"
docker build -t rl .
docker run -v $HOME/IT3105/project2:/project2 rl