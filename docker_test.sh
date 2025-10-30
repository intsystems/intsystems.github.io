# this is an extremely ineffective one-liner for testing our site
docker run  --rm -it  -v $(pwd):/data -p 4000:4000 -e LANG=C.UTF-8 -it ubuntu:20.04 /bin/bash -c "cd /data; apt-get update; apt-get install -y bundler; bundle install; bundle exec jekyll serve  --host 0.0.0.0"

