from image_utils import get_image_paths, embed_images, build_faiss_index, embed_single_image
from PIL import Image
import faiss

image_paths = get_image_paths()
embeddings = embed_images(image_paths)
index = build_faiss_index(embeddings)

query_path = "images/image1.jpg"  # change to your test image
query_image = Image.open(query_path)
query_embed = embed_single_image(query_image)
D, I = index.search(query_embed, k=3)

print("Query:", query_path)
for i in I[0]:
    print("Similar:", image_paths[i])
