#!/bin/bash

git add --all
echo $1
if [ $1 ]; then
	git commit -m $1
else
	git commit -m 'default'
fi
git push