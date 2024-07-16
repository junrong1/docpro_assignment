# Topic Classifier

## Table of Content
- [Introduction](#introduction)
- [Deployment](#deployment)
- [API Endpoint](#api-endpoint)

## Introduction
This repo is a Topic Classifier for DocPro.

## Deployment

Replace the `my-key` in the `docker-compose.yml`.

Then run `docker-compose up --build` or just `docker-compose up` if no need to rebuild. 

The service should be accessible at localhost:8080.

## API Endpoint
We offer two API endpoint `/gen_sup_topics` and `gen_unsup_topics`

The example of the `json` request
```angular2html
json
{
    "query": ["text1", "text2"]
}
```
