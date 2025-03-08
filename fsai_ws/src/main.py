import carla
import time
import numpy as np
import cv2
import pygame
from ultralytics import YOLO
import torch

class VehicleController:
    def _init_(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0

    def compute_control(self, vehicle, detected_cones):
        try:
            # Basic forward motion for testing
            self.throttle = 0.3
            self.steer = 0.0
            self.brake = 0.0
            
            return carla.VehicleControl(
                throttle=self.throttle,
                steer=self.steer,
                brake=self.brake
            )
        except Exception as e:
            print(f"Control error: {e}")
            return carla.VehicleControl()

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3]

def main():
    try:
        print("Initializing...")
        pygame.init()
        display_size = (1280, 720)
        display = pygame.display.set_mode(
            (display_size[0], display_size[1]),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Cone Detection Test")

        print("Loading YOLO model...")
        model = YOLO('/home/abdul/Documents/FSAI/runs/detect/train7/weights/best.pt')

        print("Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        # Spawn vehicle
        print("Spawning vehicle...")
        spawn_transform = carla.Transform(
            carla.Location(x=-50.0, y=0.0, z=5.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)

        # Set up camera
        print("Setting up camera...")
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '640')
        rgb_bp.set_attribute('image_size_y', '384')
        rgb_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

        controller = VehicleController()

        class SensorData:
            def _init_(self):
                self.rgb_img = None
                self.detections = None

        sensor_data = SensorData()

        def rgb_callback(image):
            try:
                rgb_image = process_image(image)
                results = model(rgb_image, verbose=False)
                if len(results) > 0:
                    sensor_data.detections = results[0].boxes.data
                sensor_data.rgb_img = rgb_image
            except Exception as e:
                print(f"Callback error: {e}")

        rgb_camera.listen(rgb_callback)
        print("Camera listening...")

        running = True
        clock = pygame.time.Clock()

        print("Entering main loop...")
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                # Apply vehicle control
                control = controller.compute_control(vehicle, sensor_data.detections)
                vehicle.apply_control(control)

                # Visualization
                if sensor_data.rgb_img is not None:
                    viz_img = sensor_data.rgb_img.copy()
                    if sensor_data.detections is not None:
                        for det in sensor_data.detections:
                            try:
                                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Detection visualization error: {e}")

                    # Display
                    viz_surface = pygame.surfarray.make_surface(
                        np.transpose(viz_img, (1, 0, 2)))
                    display.blit(viz_surface, (0, 0))
                    pygame.display.flip()

                clock.tick(20)

            except Exception as e:
                print(f"Main loop error: {e}")

    finally:
        print("Cleaning up...")
        pygame.quit()
        if 'rgb_camera' in locals():
            rgb_camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()

if __name__ == "__main__":
    main()
