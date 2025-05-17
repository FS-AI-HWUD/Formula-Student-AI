import carla
import time


def cleanup_carla_actors(host='localhost', port=2000, timeout=5.0):
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        all_actors = world.get_actors()

        removable_types = [
            'vehicle.*',
            'sensor.*'
        ]

        actors_to_destroy = []

        for filter_str in removable_types:
            actors_to_destroy.extend(all_actors.filter(filter_str))

        print(f"[INFO] Found {len(actors_to_destroy)} actors to destroy...")

        for actor in actors_to_destroy:
            try:
                actor.destroy()
                print(f"[CLEANED] Destroyed {actor.type_id} (ID: {actor.id})")
            except Exception as e:
                print(f"[WARN] Failed to destroy actor {actor.id}: {e}")

        print("[SUCCESS] Cleanup complete.")

    except Exception as e:
        print(f"[ERROR] Failed to connect or cleanup: {e}")


if __name__ == '__main__':
    cleanup_carla_actors()
