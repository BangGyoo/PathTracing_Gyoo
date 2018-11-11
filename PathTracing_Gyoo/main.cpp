#define GLM_ENABLE_EXPERIMENTAL
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include "geometry.h"

int main() {
	Vec3 x = { 0.5,0.5,0.5,1.0f };
	
	Mat4 m = Gen_Identity_Mat(1.0f);
	std::cout<<  Display_Mat4(Rotate_x(30.0f*PI/180.0f));
	//std::cout << Display_Mat4(Translate(m,x));

	//std::cout << glm::to_string(glm::rotate(glm::mat4(1.0f),0.25,x) << std::endl;

	system("pause");
	return 0;
}