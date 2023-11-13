import Metashape
import os
import re

# Define the root directory where the datasets are stored
ROOT_DIR = "E:\\Finished"

# Image Processing Functions
def align_photos(chunk):
    print("Aligning photos...")
    chunk.matchPhotos(downscale=0)
    chunk.alignCameras()
    print("Photos aligned.")

def build_depth_maps(chunk):
    print("Building depth maps...")
    chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering)
    print("Depth maps built.")

def build_mesh(chunk):
    print("Building mesh...")
    chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
    print("Mesh built.")

def texture_model(chunk):
    print("Texturing model...")
    chunk.buildUV(mapping_mode=Metashape.GenericMapping)
    chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=8192)
    print("Model textured.")

# Utility Functions
def build_dataset_list(root_dir, square_code):
    """
    Build a list of datasets to process based on the root directory and square code.
    """
    dataset_list = []
    for date_folder in sorted(os.listdir(root_dir)):
        date_folder_path = os.path.join(root_dir, date_folder)
        if os.path.isdir(date_folder_path) and re.match(r"\d{8}", date_folder):
            for square_folder in os.listdir(date_folder_path):
                if square_code in square_folder and not os.path.splitext(square_folder)[1]:
                    square_folder_path = os.path.join(date_folder_path, square_folder)
                    if os.path.isdir(square_folder_path):
                        dataset_list.append(square_folder_path)
    return dataset_list

def find_camera_by_label(chunk, label):
    """
    Find a camera in a chunk by its label.
    """
    for camera in chunk.cameras:
        if camera.label == label:
            return camera
    return None

def import_estimated_camera_positions(new_chunk, previous_chunk):
    """
    Import estimated camera positions from a previous chunk to a new chunk.
    """
    new_chunk.crs = previous_chunk.crs
    if not previous_chunk.transform.matrix:
        raise ValueError("Previous chunk has not been aligned yet.")

    for camera in previous_chunk.cameras:
        if camera.transform:
            estimated_local_position = camera.center
            estimated_global_position = previous_chunk.crs.project(previous_chunk.transform.matrix.mulp(estimated_local_position))
            new_camera = find_camera_by_label(new_chunk, camera.label)
            if new_camera:
                new_camera.reference.location = estimated_global_position
                new_camera.reference.enabled = True
            else:
                print(f"Camera {camera.label} not found in new chunk.")
    new_chunk.updateTransform()

def load_images_from_date_square(date, square_name, chunk):
    """
    Load images from a specific square subfolder for a given date into a chunk.
    """
    square_path = os.path.join(ROOT_DIR, date, f"{date}_{square_name}")
    if not os.path.isdir(square_path):
        print(f"Square folder {date}_{square_name} does not exist in path {ROOT_DIR}")
        return

    images = [os.path.join(square_path, img) for img in os.listdir(square_path) 
              if os.path.isfile(os.path.join(square_path, img)) and img.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png'))]
    
    if images:
        chunk.addPhotos(images)
        for camera in chunk.cameras:
            camera.reference.enabled = False
    else:
        print(f"No images found in {square_path}")

def process_new_chunk(chunk, square_name, is_transfer_georeferencing, last_chunk):
    """
    Process a new chunk based on whether it's for georeferencing transfer or a regular date processing.
    """
    if is_transfer_georeferencing:
        transfer_georeferencing(chunk, square_name)
    else:
        process_date(chunk, square_name)

    import_estimated_camera_positions(chunk, last_chunk)
    align_photos(chunk)

    if not is_transfer_georeferencing:
        build_depth_maps(chunk)
        build_mesh(chunk)
        texture_model(chunk)

def transfer_georeferencing(chunk, square_name):
    """
    Transfer georeferencing for a chunk.
    """
    dates = chunk.label.split('_')
    for date in dates:
        load_images_from_date_square(date, square_name, chunk)

def process_date(chunk, square_name):
    """
    Process a single date chunk.
    """
    load_images_from_date_square(chunk.label, square_name, chunk)

def get_next_date_path(current_date):
    """
    Get the path of the next date folder after the given date.
    """
    date_folders = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)) and re.match(r"\d{8}", d)])
    current_index = date_folders.index(current_date) if current_date in date_folders else -1
    return os.path.join(ROOT_DIR, date_folders[current_index + 1]) if current_index + 1 < len(date_folders) else None

def main():
    doc = Metashape.app.document
    project_name = os.path.basename(doc.path)
    square_name, _ = os.path.splitext(project_name)

    while True:
        sorted_chunks = sorted(doc.chunks, key=lambda c: c.label)
        last_chunk = sorted_chunks[-1]
        last_chunk_dates = last_chunk.label.split('_')

        if len(last_chunk_dates) == 2:
            new_chunk_date = last_chunk_dates[1]
            new_chunk_label = new_chunk_date
            is_transfer_georeferencing = False
        else:
            next_date_path = get_next_date_path(last_chunk_dates[0])
            if next_date_path:
                next_date = os.path.basename(next_date_path)
                new_chunk_label = f"{last_chunk.label}_{next_date}"
                is_transfer_georeferencing = True
            else:
                print("No newer date available for processing.")
                break

        new_chunk = doc.addChunk()
        new_chunk.label = new_chunk_label
        process_new_chunk(new_chunk, square_name, is_transfer_georeferencing, last_chunk)
        doc.save()

if __name__ == "__main__":
    main()
