
#define DEBUG


#ifndef GPU_KERNEL			// 최초 cl 코드 앞에 선언 할 definition
#include <math.h>
#endif

#ifdef DEBUG			// cl 코드는 호환 안되므로 꺼야함
#include <string>
#endif


///////////////////// data structure ////////////////////////////////

typedef struct _Point {
	float x;
	float y;
	float z;
	float w;
}Point;
typedef Point Vec3;		// 선언은 Init를 거쳐서 만들자.

Vec3 InitVec3(float x, float y, float z) {
	Vec3 v = { x,y,z,1.0f };
	return v;
}
Point InitPoint(float x, float y, float z) {
	Point p = { x,y,z,0.0f };
	return p;
}



union Mat4 {	// 마지막 v[3]인자는 항상 v(0,0,0,1) 로!!
	Vec3 v[4];			// union은 선언 순서에 따라 최초 필요 정의 순서도 바뀐다.
	float e[4][4];		// ex float[4][4]가 먼저 오면 정의때도 float 먼저 써야한다.
						// 그리고 항상 초기화 해줘야한다.
};
Mat4 Gen_Identity_Mat(float arg){
	Mat4 result = { arg ,0  ,0,  0,
					0, arg ,0,  0,
					0,  0, arg, 0,
					0,  0,  0, arg };
	return result;
};

///////////////////////////////////// Debug monitoring /////////////////
#ifdef DEBUG
std::string Display_Vec3(Vec3 v) {
	return std::string("x = " + std::to_string(v.x) +
		" , y = " + std::to_string(v.y) +
		" , z " + std::to_string(v.z));
}
bool Is_W_1(Vec3 v) {
	if (v.w != 1.0f) return false;
	else true;
}
Mat4 Generator1(float arg) {
	Mat4 result = { arg, arg, arg, arg,
	arg, arg, arg, arg, 
	arg, arg, arg, arg, 
	arg, arg, arg, arg};
	return result;
}
Mat4 Generator2(float arg) {
	Mat4 result = {arg ,0  ,0,  0,
					0, arg ,0,  0,
					0,  0, arg, 0,
					0,  0,  0, arg };
	return result;
}
#endif 

///////////////////////////////////// function ///////////////////////
Vec3 Add_Vec3(Vec3 v1,Vec3 v2){
	Vec3 result = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
	return result;
}
Vec3 Sub_Vec3(Vec3 v1, Vec3 v2) {
	Vec3 result = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
	return result;
}
Vec3 Mul_Vec3(float scala, Vec3 v) {
	Vec3 result = { scala * v.x , scala * v.y , scala * v.z };
	return result;
}
Vec3 Div_Vec3(float scala, Vec3 v) {
	Vec3 result = { v.x / scala , v.y / scala , v.z / scala };
	return result;
}
float Dot(Vec3 v1, Vec3 v2) {
	float result = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	return result;
}

Vec3 Cross(Vec3 v1, Vec3 v2) {
	Vec3 result = { v1.y * v2.z - v1.z * v2.y,
					v1.z * v2.x - v1.x * v2.z,
					v1.x * v2.y - v1.y * v2.x
	};
	return result;
}

#ifdef GPU_KERNEL			// CL code
float Is_Length_Of(Vec3 v) {
	return sqrt(v.x*v.x + v.y * v.y + v.z * v.z);
}
#else
float Is_Length_Of(Vec3 v) {
	return sqrt(v.x*v.x + v.y * v.y + v.z * v.z);
}
#endif

Mat4 Add_Mat4(Mat4 m1, Mat4 m2) {
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++) m1.e[i][j] += m2.e[i][j];
	return m1;
}
Mat4 Sub_mat4(Mat4 m1, Mat4 m2) {
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) m1.e[i][j] -= m2.e[i][j];
	return m1;
}
Mat4 Mul_mat4(Mat4 m1, Mat4 m2) {
	Mat4 result = { 0.0f, };
	int i, j,k;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			result.e[i][j] = 0;
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				result.e[i][j] += (m1.e[i][k] * m2.e[k][j]);
			}
		}
	}
	return result;
}

Vec3 Mul_MatVec(Mat4 m, Vec3 v) {
	v.x = m.e[0][0] * v.x + m.e[0][1] * v.y + m.e[0][2] * v.z + m.e[0][3] * v.w;
	v.y = m.e[1][0] * v.x + m.e[1][1] * v.y + m.e[1][2] * v.z + m.e[1][3] * v.w;
	v.z = m.e[2][0] * v.x + m.e[2][1] * v.y + m.e[2][2] * v.z + m.e[2][3] * v.w;
	v.w = m.e[3][0] * v.x + m.e[3][1] * v.y + m.e[3][2] * v.z + m.e[3][3] * v.w;
	return v;
}

Mat4 Translate(Mat4 m, Vec3 v) {
	m.v[0].w += v.x;
	m.v[1].w += v.y;
	m.v[2].w += v.z;
	return m;
}
Mat4 Scale(Mat4 m, Vec3 v) {
	m.v[0].x *= v.x;
	m.v[1].y *= v.y;
	m.v[2].z *= v.z;
	return m;
}

/*rotate 부터 만들자 x,y,z 축으로 따로따로 만들어줘야할 듯하다.*/
