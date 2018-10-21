install:
	yarn install
	stack install

build:
	yarn run build
	blog build

clean:
	blog clean

watch:
	blog watch
