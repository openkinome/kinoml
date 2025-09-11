### Building the docker image

To build the Docker image, run the following command in the root
directory of the project:

``` bash
docker build -t kinoml:latest -f devtools/docker/Dockerfile .
```

To use the existing docker image, run the following command.

``` bash
docker run -it --rm \
    -p 8888:8888 \
    openkinome/kinoml:latest
```

If you have a valid OpenEye license, make sure you have set the
`$OE_LICENSE` correctly on the host machine.

``` bash
docker run -it --rm \
    -p 8888:8888 \
    -e OE_LICENSE=/mnt/oe_license.txt \
    -v ${OE_LICENSE}:/mnt/oe_license.txt:ro \
    openkinome/kinoml:latest
```

