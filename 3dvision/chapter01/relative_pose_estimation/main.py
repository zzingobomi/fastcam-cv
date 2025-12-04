from project_root_finder import DATA_DIR
import matcher


if __name__ == "__main__":
    src_path = DATA_DIR / "phototourism" / "british_museum" / "00350405_2611802704.jpg"
    dst_path = DATA_DIR / "phototourism" / "british_museum" / "01858319_78150445.jpg"

    src_kpts, dst_kpts = matcher.compute_correspondence_matching(src_path, dst_path)
    matcher.visualize(src_path, dst_path, src_kpts, dst_kpts)