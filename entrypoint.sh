
#!/bin/bash

echo "DOC_GIT_REPO  : $DOC_GIT_REPO"
echo "DOC_LOCATION  : $DOC_LOCATION"
echo "TEMP_DIR      : $TEMP_DIR"

if [ -z ${DOC_GIT_REPO+x} ]; then
    echo "Provide GIT repository location"
    exit 1
fi
if [ -z ${DOC_GIT_REPO+x} ]; then
    echo "Document location is not set. Provide location inside directory"
    exit 1
fi

mkdir ${TEMP_DIR}/source_repo
git clone ${DOC_GIT_REPO} ${TEMP_DIR}/source_repo

python3 -u  ./Langchain-Redis-Ingest.py