#include "fluids.hpp"

// Start of main
int main(int argc, char *argv[]) 
{
  
  // Check argc 
  if (argc != 3) {
    std::cerr << "Invalid Input count" << std::endl;
    std::cerr << "Usage: [particle count] [cube size]" << std::endl;
    return 1;
  }

  // Set command line inputs 
  int particle_count = static_cast<int>(flds::Context::parse(argv, 1));
  float cube_size = flds::Context::parse(argv, 2);

  // std::cout << "Particle Count: " << particle_count << "\nCube Size: " << cube_size << std::endl;

  // Start context
  flds::Context context_(cube_size);
  // std::cout << "Context Created" << std::endl;

  // Local variables
  SDL_Event event;
  glm::vec3 center(cube_size / 2.0, cube_size / 2.0, cube_size / 2.0), position(1.0);
  glm::vec3 view_position = glm::vec3(cube_size / 2.0, cube_size / 2.0, cube_size * 3.0);
  glm::mat4 model(1.0), cube_model(1.0), sphere_model(1.0);
  bool lmb_down = false, initial_mouse = true;
  
  // Set initial values
  float yaw = 0.0, pitch = 0.0, fov = 60.0;
  float near_plane = 0.1, far_plane = 100.0, aspect_ratio = 1280.0 / 720.0;
  float last_mouse_x = 0.0, last_mouse_y = 0.0, x_offset = 0.0, y_offset = 0.0, sensititivity = 5e-5;
  float particle_spacing = cube_size / cbrt(particle_count), h;
  float radius;

  // Set the smoothingRadius and model radius 
  h = (particle_spacing < 0.8) ? particle_spacing * alpha : 1.2; 
  radius = 0.2 * h;
  
  // Create shape meshes
  mesh::Sphere_GL sphere_gl(40, 40, radius);
  mesh::Cube_GL cube_gl(cube_size);

  // Initialize the simulation 
  startSim(cube_size, particle_count, h);

  // Update loop
  while (true) {
    // Clear color
    glClearColor(0.0, 0.0, 0.0, 1.0);

    // std::cout << "Clear Color" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // std::cout << "Pre-Iterator" << std::endl;

    // Update Position State
    particleIterator(particle_count);

    // std::cout << "Iterated over particles" << std::endl;

    // Polls for events in SDL instance
    while (SDL_PollEvent(&event)) {
      // Quit program out
      if (event.type == SDL_QUIT) {
        exit(0);
      }
      // Checks for specific keypresses
      switch (event.type) {
        case SDL_KEYDOWN:
          // Handle Key events
          switch (event.key.keysym.sym) {
            // Escape exit
            case SDLK_ESCAPE:
              exit(0);
              break;
            // FOV sliders c and v keys
            case SDLK_c:
              fov += 1;
              break;
            case SDLK_v:
              fov -= 1;
              break;
            default:
              break;
          }
        case SDL_MOUSEBUTTONDOWN:
          if (event.button.button == SDL_BUTTON_LEFT) {
            lmb_down = true;
          }
          break;
        case SDL_MOUSEBUTTONUP:
          if (event.button.button == SDL_BUTTON_LEFT) {
            lmb_down = false;
            initial_mouse = true;
          }
          break;
        // Handle Moving Mouse
        case SDL_MOUSEMOTION:
          // Only handle if left mouse down is true
          if (lmb_down) {
            // Set initial mouse position
            if (initial_mouse) {
              last_mouse_x = event.motion.x;
              last_mouse_y = event.motion.y;
              initial_mouse = false;
            }

            // Calculate Mouse offsets
            x_offset = (event.motion.x - last_mouse_x) * sensititivity;
            y_offset = (event.motion.y - last_mouse_y) * sensititivity;

            // Find yaw and pitch 
            yaw -= x_offset;
            pitch += y_offset;
            pitch = (pitch > 89.0) ? 89.0 : pitch;
            pitch = (pitch < -89.0) ? -89.0 : pitch;

            // Set View Matrix
            cube_model = glm::translate(model, center);
            cube_model = glm::rotate(cube_model, yaw, glm::vec3(0.0, 1.0, 0.0));
            cube_model = glm::rotate(cube_model, pitch, glm::vec3(1.0, 0.0, 0.0));
            cube_model = glm::translate(cube_model, -center);
          }
          break;
        // Default case
        default:
          break;
      }
    }

    // std::cout << "Pre-render Call" << std::endl;

    context_.shader_->Render(fov, aspect_ratio, near_plane, far_plane, cube_gl.object_color_, cube_model);
    cube_gl.DrawCube();

    // std::cout << "Cube Draw Call" << std::endl;

    for (int particle = 0; particle < particle_count; ++particle) {
      position = glm::vec3(
        particles[particle].position.data[0],
        particles[particle].position.data[1],
        particles[particle].position.data[2]
      );

      sphere_model = glm::translate(cube_model, position);

      context_.shader_->camera_.lighting_.SetPosition(view_position + glm::vec3(0.0, 1.0, 0.0));
      context_.shader_->Render(fov, aspect_ratio, near_plane, far_plane, sphere_gl.object_color_, sphere_model);
      sphere_gl.DrawSphere();
    }

    // std::cout << "Sphere Draw Call" << std::endl;

    // Swap buffers
    SDL_GL_SwapWindow(context_.window_);
  }

  // Free memory
  freeTables();
  free(particles);

  return 0;
}
// End of main