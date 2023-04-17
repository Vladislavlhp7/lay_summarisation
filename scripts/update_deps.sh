#!/bin/bash

poetry lock
poetry export -f requirements.txt --without-hashes > requirements.txt