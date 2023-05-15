tokenize=${1:-"1"}
preprocess=${2:-"1"}
train=${3:-"1"}
eval=${4:-"1"}

if [ ${tokenize} = "1" ]; then
    echo "================tokenize================"
fi

if [ ${preprocess} = "1" ]; then
    echo "================preprocess================"
fi

if [ ${train} = "1" ]; then
    echo "================train================"
fi

if [ ${eval} = "1" ]; then
    echo "================evaluate================"
fi