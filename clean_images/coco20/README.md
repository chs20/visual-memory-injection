20 random images from coco val 2017


src_dir="path/to/source"
dst_dir="path/to/destination"
n=10

mkdir -p "$dst_dir"
shuf -n "$n" <(find "$src_dir" -maxdepth 1 -type f -name '*.jpg') \
  | xargs -I{} cp {} "$dst_dir"
