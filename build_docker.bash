#!/bin/bash
#

PACKAGE_NAME=mlproject
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAUT_DOCKERFILE=${ROOT_DIR}/Dockerfile

# Function to display help
show_help() {
  echo "Usage: $0 --dockerfile <path_to_dockerfile>"
  echo
  echo "Options:"
  echo "  --dockerfile PATH   Specify the path to the Dockerfile"
  echo "  -h, --help          Display this help message"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dockerfile)
      DOCKERFILE="$2"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
  shift
done


if [[ -z "$DOCKERFILE" ]]; then
  echo "--dockerfile flag was not given, using default: $DEFAUT_DOCKERFILE"
  DOCKERFILE=$DEFAUT_DOCKERFILE
fi

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Error: Dockerfile not found at path '$DOCKERFILE'"
  exit 1
fi

docker build -f ${DOCKERFILE} --build-arg UID=$(id -u) \
             --build-arg GID=$(id -g) \
             --build-arg USERNAME=$USER \
             --build-arg GROUPNAME=$(id -gn) \
            --build-arg PACKAGE_NAME=${PACKAGE_NAME} \
             -t $PACKAGE_NAME ${ROOT_DIR}
