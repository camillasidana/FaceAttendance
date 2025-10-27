import os
import cv2


def verify_dataset():
    dataset_path = '../data/raw_images'

    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        return

    print("=" * 50)
    print("📊 DATASET VERIFICATION")
    print("=" * 50)

    people = os.listdir(dataset_path)

    if not people:
        print("❌ No person folders found in known_faces!")
        return

    total_images = 0

    for person in people:
        person_folder = os.path.join(dataset_path, person)

        if os.path.isdir(person_folder):
            images = [f for f in os.listdir(person_folder)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            total_images += image_count

            print(f"\n👤 Person: {person}")
            print(f"   📸 Images: {image_count}")

            if image_count < 10:
                print(f"   ⚠️  WARNING: Only {image_count} images. Recommend at least 10-15 images.")
            else:
                print(f"   ✅ Good! Sufficient images for training.")

    print("\n" + "=" * 50)
    print(f"✅ Total people: {len(people)}")
    print(f"✅ Total images: {total_images}")
    print("=" * 50)

    # Show sample images
    show_sample = input("\nDo you want to see sample images? (y/n): ").lower()

    if show_sample == 'y':
        for person in people:
            person_folder = os.path.join(dataset_path, person)
            if os.path.isdir(person_folder):
                images = [f for f in os.listdir(person_folder)
                          if f.endswith(('.jpg', '.jpeg', '.png'))]

                if images:
                    # Show first image
                    img_path = os.path.join(person_folder, images[0])
                    img = cv2.imread(img_path)

                    if img is not None:
                        cv2.imshow(f'Sample: {person} - Press any key for next', img)
                        cv2.waitKey(0)

        cv2.destroyAllWindows()

    print("\n✅ Dataset verification complete!")


if __name__ == "__main__":
    verify_dataset()