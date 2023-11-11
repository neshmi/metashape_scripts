import Metashape
import os
import re

# Define the root directory where the datasets are stored
root_dir = "E:\\Finished"

def align_photos(chunk):
    print("Aligning photos...")
    chunk.matchPhotos(downscale=0)
    chunk.alignCameras()
    print("Photos aligned.")

def build_depth_maps(chunk):
    print("Building depth maps...")
    chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering)
    print("Depth maps built.")

def build_dense_cloud(chunk):
    print("Building dense cloud...")
    chunk.buildPointCloud(point_confidence=True)
    print("Dense cloud built.")

def build_mesh(chunk):
    print("Building mesh...")
    chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
    print("Mesh built.")

def texture_model(chunk):
    print("Texturing model...")
    chunk.buildUV(mapping_mode=Metashape.GenericMapping)
    chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=8192)
    print("Model textured.")


# Function to build a list of datasets to process
def build_dataset_list(root_dir, square_code):
    dataset_list = []
    # Iterate over all items in the root directory
    for item in sorted(os.listdir(root_dir)):
        date_folder_path = os.path.join(root_dir, item)
        # Check if the item is a folder and follows the date format
        if os.path.isdir(date_folder_path) and re.match(r"\d{8}", item):
            # Check for subfolders that contain the square code and do not have an extension
            for subitem in os.listdir(date_folder_path):
                if square_code in subitem and not os.path.splitext(subitem)[1]:
                    square_folder_path = os.path.join(date_folder_path, subitem)
                    # Check if the square folder path is indeed a directory
                    if os.path.isdir(square_folder_path):
                        dataset_list.append(square_folder_path)  # Append the full path
    return dataset_list

# Function to extract the position for a specific camera from a given chunk
def extract_camera_position(camera, chunk):
    if not chunk.transform.matrix or not camera.transform:
        raise ValueError("Chunk has not been aligned or camera has no valid transformation.")
    position = chunk.transform.matrix.mulp(camera.center)  # Convert to chunk coordinate system
    orientation = camera.transform.rotation()
    camera_info = {
        'label': camera.label,
        'x': position.x,
        'y': position.y,
        'z': position.z,
        'yaw': orientation.yaw,
        'pitch': orientation.pitch,
        'roll': orientation.roll
    }
    return camera_info

def find_camera_by_label(chunk, label):
    for camera in chunk.cameras:
        if camera.label == label:
            return camera
    return None



def import_estimated_camera_positions(new_chunk, previous_chunk):
    # Ensure the previous chunk has been aligned
    new_chunk.crs = previous_chunk.crs
    if not previous_chunk.transform.matrix:
        raise ValueError("Previous chunk has not been aligned yet.")

    # Loop through each camera in the previous chunk
    for camera in previous_chunk.cameras:
        # Check if the camera has an estimated position
        if camera.transform:
            # Get the estimated position in the local coordinate system
            estimated_local_position = camera.center
            # Convert the local position to the global coordinate system
            estimated_global_position = previous_chunk.crs.project(previous_chunk.transform.matrix.mulp(estimated_local_position))

            # Now, apply this estimated global position to the new camera in the new chunk
            new_camera = find_camera_by_label(new_chunk, camera.label)
            if new_camera:
                new_camera.label = camera.label
                new_camera.reference.location = estimated_global_position
                new_camera.reference.enabled = True  # Enable the reference for alignment
            else:
                print(f"Camera {camera.label} not found in new chunk.")
    # Update the chunk transform after importing camera positions
    new_chunk.updateTransform()



# Function to get the next date folder path after the given date
def get_next_date_path(root_dir, current_date):
    date_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r"\d{8}", d)])
    current_index = date_folders.index(current_date) if current_date in date_folders else -1
    # Return the next date folder path if available
    return os.path.join(root_dir, date_folders[current_index + 1]) if current_index + 1 < len(date_folders) else None

# Function to load images from a specific square subfolder for a given date
def load_images_from_date_square(date, square_name, chunk):
    # Construct the square path with the date and square name
    images = []
    date_path = os.path.join(root_dir, f"{date}")
    square_path = os.path.join(date_path, f"{date}_{square_name}")
    if not os.path.isdir(square_path):
        print(f"Square folder {date}_{square_name} does not exist in path {root_dir}")
        return

    # Load all images from the square subfolder
    for image_name in os.listdir(square_path):
        image_path = os.path.join(square_path, image_name)
        if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):
            images.append(image_path)

    chunk.addPhotos(images)
    for camera in chunk.cameras:
        camera.reference.enabled = False

def transfer_georeferencing(chunk, square_name):
    print(f"Transferring georeferencing for chunk {chunk.label}")
    # Extract the two dates from the chunk label
    dates = chunk.label.split('_')

    # Load all images from the square subfolder for the first date
    load_images_from_date_square(dates[0], square_name, chunk)

    # Load all images from the square subfolder for the second date
    load_images_from_date_square(dates[1], square_name, chunk)

    pass



# Function to process a single date chunk
def process_date(chunk, square_name):
    # Implement the logic to process a single date chunk
    date = chunk.label
    load_images_from_date_square(date, square_name, chunk)

    pass

# Main processing function
def main():
    doc = Metashape.app.document
    project_name = os.path.basename(doc.path)
    square_name, _ = os.path.splitext(project_name)

    while True:
        # Sort chunks by label assuming they have the date in their label
        sorted_chunks = sorted(doc.chunks, key=lambda c: c.label)

        # Identify the last chunk (most recent date)
        last_chunk = sorted_chunks[-1]
        last_chunk_dates = last_chunk.label.split('_')

        if len(last_chunk_dates) == 2:
            # Process the second date in the two-date chunk
            new_chunk_date = last_chunk_dates[1]
            new_chunk_label = new_chunk_date
            is_transfer_georeferencing = False
        else:
            # Find the next date available
            next_date_path = get_next_date_path(root_dir, last_chunk_dates[0])
            if next_date_path:
                next_date = os.path.basename(next_date_path)
                new_chunk_label = last_chunk.label + "_" + next_date
                is_transfer_georeferencing = True
            else:
                print("No newer date available for processing.")
                break  # Exit the loop if there is no next date

        # Create and process the new chunk
        new_chunk = doc.addChunk()
        new_chunk.label = new_chunk_label
        if is_transfer_georeferencing:
            transfer_georeferencing(new_chunk, square_name)
            new_chunk.crs = last_chunk.crs
            import_estimated_camera_positions(new_chunk, last_chunk)
            align_photos(new_chunk)
        else:
            process_date(new_chunk, square_name)
            import_estimated_camera_positions(new_chunk, last_chunk)
            align_photos(new_chunk)
            build_depth_maps(new_chunk)
            build_mesh(new_chunk)
            texture_model(new_chunk)

        # Save the document after updating the chunks
        doc.save()

if __name__ == "__main__":
    main()
