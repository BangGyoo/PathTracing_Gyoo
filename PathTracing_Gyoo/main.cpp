#define GLM_ENABLE_EXPERIMENTAL
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include "geometry.h"

int main() {
	glm::vec3 x(0.5f);
	
	std::cout << glm::to_string(glm::scale(glm::mat4(1.0f),x)) << std::endl;

	system("pause");
	return 0;
}