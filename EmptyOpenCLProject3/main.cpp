#define GLM_ENABLE_EXPERIMENTAL
#include "geometry.h"
#include "Utilities.h"

#include <omp.h>
int winID;
#define _USE_MATH_DEFINES
#define GI
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <queue>
#include <random>
#include <glm/glm/glm.hpp>
#include <glm/glm/gtx/string_cast.hpp>
#include <Assimp/include/assimp/Importer.hpp>  
#include <Assimp/include/assimp/scene.h>
#include <Assimp/include/assimp/postprocess.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iterator>
#define M_PI 3.141592653589793 


std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}
using namespace std;
static const float kInfinity = std::numeric_limits<float>::max();
static const float kEpsilon = 1e-8;
static const Vec3f kDefaultBackgroundColor = Vec3f(0.235294, 0.67451, 0.843137);
template <> const Matrix44f Matrix44f::kIdentity = Matrix44f();

inline
float clamp(const float &lo, const float &hi, const float &v)
{
	return std::max(lo, std::min(hi, v));
}

inline
float deg2rad(const float &deg)
{
	return deg * M_PI / 180;
}

inline
Vec3f mix(const Vec3f &a, const Vec3f& b, const float &mixValue)
{
	return a * (1 - mixValue) + b * mixValue;
}

struct Options
{
	uint32_t width = 640;
	uint32_t height = 480;
	float fov = 90;
	Vec3f backgroundColor = kDefaultBackgroundColor;
	Matrix44f cameraToWorld;
	float bias = 0.0001;
	uint32_t maxDepth = 1;
};

enum MaterialType { kDiffuse };

class Object
{
public:
	// [comment]
	// Setting up the object-to-world and world-to-object matrix
	// [/comment]
	Object(const Matrix44f &o2w) : objectToWorld(o2w), worldToObject(o2w.inverse()) {}
	virtual ~Object() {}
	virtual bool intersect(const Vec3f &, const Vec3f &, float &, uint32_t &, Vec2f &) const = 0;
	virtual void getSurfaceProperties(const Vec3f &, const Vec3f &, const uint32_t &, const Vec2f &, Vec3f &, Vec2f &) const = 0;
	Matrix44f objectToWorld, worldToObject;
	MaterialType type = kDiffuse;
	Vec3f albedo = 0.18;
	float Kd = 0.8; // phong model diffuse weight
	float Ks = 0.2; // phong model specular weight
	float n = 3;   // phong specular exponent
};

// [comment]
// Compute the roots of a quadratic equation
// [/comment]
bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
	float discr = b * b - 4 * a * c;
	if (discr < 0) return false;
	else if (discr == 0) {
		x0 = x1 = -0.5 * b / a;
	}
	else {
		float q = (b > 0) ?
			-0.5 * (b + sqrt(discr)) :
			-0.5 * (b - sqrt(discr));
		x0 = q / a;
		x1 = c / q;
	}

	return true;
}

// [comment]
// Sphere class. A sphere type object
// [/comment]
class Sphere : public Object
{
public:
	Sphere(const Matrix44f &o2w, const float &r) : Object(o2w), radius(r), radius2(r *r)
	{
		o2w.multVecMatrix(Vec3f(0), center);
	}
	// [comment]
	// Ray-sphere intersection test
	// [/comment]
	bool intersect(
		const Vec3f &orig,
		const Vec3f &dir,
		float &tNear,
		uint32_t &triIndex, // not used for sphere
		Vec2f &uv) const    // not used for sphere
	{
		float t0, t1; // solutions for t if the ray intersects
					  // analytic solution
		Vec3f L = orig - center;
		float a = dir.dotProduct(dir);
		float b = 2 * dir.dotProduct(L);
		float c = L.dotProduct(L) - radius2;
		if (!solveQuadratic(a, b, c, t0, t1)) return false;

		if (t0 > t1) std::swap(t0, t1);

		if (t0 < 0) {
			t0 = t1; // if t0 is negative, let's use t1 instead
			if (t0 < 0) return false; // both t0 and t1 are negative
		}

		tNear = t0;

		return true;
	}
	// [comment]
	// Set surface data such as normal and texture coordinates at a given point on the surface
	// [/comment]
	void getSurfaceProperties(
		const Vec3f &hitPoint,
		const Vec3f &viewDirection,
		const uint32_t &triIndex,
		const Vec2f &uv,
		Vec3f &hitNormal,
		Vec2f &hitTextureCoordinates) const
	{
		hitNormal = hitPoint - center;
		hitNormal.normalize();
		// In this particular case, the normal is simular to a point on a unit sphere
		// centred around the origin. We can thus use the normal coordinates to compute
		// the spherical coordinates of Phit.
		// atan2 returns a value in the range [-pi, pi] and we need to remap it to range [0, 1]
		// acosf returns a value in the range [0, pi] and we also need to remap it to the range [0, 1]
		hitTextureCoordinates.x = (1 + atan2(hitNormal.z, hitNormal.x) / M_PI) * 0.5;
		hitTextureCoordinates.y = acosf(hitNormal.y) / M_PI;
	}
	float radius, radius2;
	Vec3f center;
};

bool rayTriangleIntersect(
	const Vec3f &orig, const Vec3f &dir,
	const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
	float &t, float &u, float &v)
{
	Vec3f v0v1 = v1 - v0;		// normal cpu version
	Vec3f v0v2 = v2 - v0;
	Vec3f pvec = dir.crossProduct(v0v2);
	float det = v0v1.dotProduct(pvec);

	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < kEpsilon) return false;

	float invDet = 1 / det;

	Vec3f tvec = orig - v0;
	u = tvec.dotProduct(pvec) * invDet;
	if (u < 0 || u > 1) return false;

	Vec3f qvec = tvec.crossProduct(v0v1);
	v = dir.dotProduct(qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = v0v2.dotProduct(qvec) * invDet;

	return (t > 0) ? true : false;
	
}

cl_float3 &Trans_float3(const Vec3f &v) {
	cl_float3 result;
	result.x = v.x;
	result.y = v.y;
	result.z = v.z;

	return result;
}
typedef struct _cl_tri {
	cl_float3 v0;
	cl_float3 v1;
	cl_float3 v2;
}cl_tri;

void Init_CL();
cl::CommandQueue *CL_queue;
std::vector<cl::Device> devices;
cl::Program::Sources *sources;
std::vector<cl::Platform> platforms;
cl::Context *context;
cl::Device device;
cl::Program *program;
std::string *kernel_code;
clock_t start;
class TriangleMesh : public Object
{
public:
	// Build a triangle mesh from a face index array and a vertex index array
	TriangleMesh(
		const Matrix44f &o2w,
		const uint32_t nfaces,
		const std::unique_ptr<uint32_t[]> &faceIndex,
		const std::unique_ptr<uint32_t[]> &vertsIndex,
		const std::unique_ptr<Vec3f[]> &verts,
		std::unique_ptr<Vec3f[]> &normals,
		std::unique_ptr<Vec2f[]> &st) :
		Object(o2w),
		numTris(0)
	{
		uint32_t k = 0, maxVertIndex = 0;
		// find out how many triangles we need to create for this mesh
		for (uint32_t i = 0; i < nfaces; ++i) {
			numTris += faceIndex[i] - 2;
			for (uint32_t j = 0; j < faceIndex[i]; ++j)
				if (vertsIndex[k + j] > maxVertIndex)
					maxVertIndex = vertsIndex[k + j];
			k += faceIndex[i];
		}
		maxVertIndex += 1;

		// allocate memory to store the position of the mesh vertices
		P = std::unique_ptr<Vec3f[]>(new Vec3f[maxVertIndex]);
		for (uint32_t i = 0; i < maxVertIndex; ++i) {
			// [comment]
			// Transforming vertices to world space
			// [/comment]
			objectToWorld.multVecMatrix(verts[i], P[i]);
		}

		// allocate memory to store triangle indices
		trisIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
		uint32_t l = 0;
		N = std::unique_ptr<Vec3f[]>(new Vec3f[numTris * 3]);
		sts = std::unique_ptr<Vec2f[]>(new Vec2f[numTris * 3]);
		// [comment]
		// Computing the transpse of the object-to-world inverse matrix
		// [/comment]
		Matrix44f transformNormals = worldToObject.transpose();
		// generate the triangle index array and set normals and st coordinates
		for (uint32_t i = 0, k = 0; i < nfaces; ++i) { // for each  face
			for (uint32_t j = 0; j < faceIndex[i] - 2; ++j) { // for each triangle in the face
				trisIndex[l] = vertsIndex[k];
				trisIndex[l + 1] = vertsIndex[k + j + 1];
				trisIndex[l + 2] = vertsIndex[k + j + 2];
				// [comment]
				// Transforming normals
				// [/comment]
				transformNormals.multDirMatrix(normals[k], N[l]);
				transformNormals.multDirMatrix(normals[k + j + 1], N[l + 1]);
				transformNormals.multDirMatrix(normals[k + j + 2], N[l + 2]);
				N[l].normalize();
				N[l + 1].normalize();
				N[l + 2].normalize();
				sts[l] = st[k];
				sts[l + 1] = st[k + j + 1];
				sts[l + 2] = st[k + j + 2];
				l += 3;
			}
			k += faceIndex[i];
		}
	}
	// Test if the ray interesests this triangle mesh
	bool intersect(const Vec3f &orig, const Vec3f &dir, float &tNear, uint32_t &triIndex, Vec2f &uv) const
	{
		
		uint32_t j = 0;
		bool isect = false;
		for (uint32_t i = 0; i < numTris; ++i) {
			const Vec3f &v0 = P[trisIndex[j]];
			const Vec3f &v1 = P[trisIndex[j + 1]];
			const Vec3f &v2 = P[trisIndex[j + 2]];
			float t = kInfinity, u, v;

			//Sleep(2000);
			if (rayTriangleIntersect(orig, dir, v0, v1, v2, t, u, v) && t < tNear) {
				tNear = t;
				uv.x = u;
				uv.y = v;
				triIndex = i;
				isect = true;
			}
			j += 3;
		}
		return isect;
		/*
		if(numTris > 500) start = clock();
		cl_float3 *cl_v0 = new cl_float3[numTris];
		cl_float3 *cl_v1 = new cl_float3[numTris];
		cl_float3 *cl_v2 = new cl_float3[numTris];
		cl_float2 *cl_uv = new cl_float2[numTris];
		cl_bool *result = new cl_bool[numTris];
		cl_float *cl_tNear = new cl_float[numTris];

		for (int i = 0; i < numTris; i++) {
			cl_v0[i] = Trans_float3(P[trisIndex[3 * i]]);
			cl_v1[i] = Trans_float3(P[trisIndex[3 * i + 1]]);
			cl_v2[i] = Trans_float3(P[trisIndex[3 * i + 2]]);
		}
		
		for (int i = 0; i < numTris; i++) cl_tNear[i] = tNear;
		cl_float3 cl_orig = Trans_float3(orig);
		cl_float3 cl_dir = Trans_float3(dir);
		cl::Buffer buffer_v0(*context, CL_MEM_READ_WRITE, sizeof(cl_float3)*numTris);
		cl::Buffer buffer_v1(*context, CL_MEM_READ_WRITE, sizeof(cl_float3)*numTris);
		cl::Buffer buffer_v2(*context, CL_MEM_READ_WRITE, sizeof(cl_float3)*numTris);
		cl::Buffer buffer_tNear(*context, CL_MEM_READ_WRITE, sizeof(cl_float)*numTris);
		cl::Buffer buffer_uv(*context, CL_MEM_READ_WRITE, sizeof(cl_float2)*numTris);
		cl::Buffer buffer_result(*context, CL_MEM_READ_WRITE, sizeof(cl_bool)*numTris);
		program = new cl::Program(*context, *sources);
		if (program->build({ device }) != CL_SUCCESS) {
			std::cout << "Error building: " << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
			exit(1);
		}
		cl::CommandQueue CL_queue(*context, device, 0, NULL);
		CL_queue.enqueueWriteBuffer(buffer_v0, CL_TRUE, 0, sizeof(cl_float3)*numTris, cl_v0);
		CL_queue.enqueueWriteBuffer(buffer_v1, CL_TRUE, 0, sizeof(cl_float3)*numTris, cl_v1);
		CL_queue.enqueueWriteBuffer(buffer_v2, CL_TRUE, 0, sizeof(cl_float3)*numTris, cl_v2);
		CL_queue.enqueueWriteBuffer(buffer_tNear, CL_TRUE, 0, sizeof(cl_float)*numTris, cl_tNear);
		cl::Kernel Intersect_mesh(*program, "Intersect_mesh");
		Intersect_mesh.setArg(0, buffer_v0);
		Intersect_mesh.setArg(1, buffer_v1);
		Intersect_mesh.setArg(2, buffer_v2);
		Intersect_mesh.setArg(3, buffer_tNear);
		Intersect_mesh.setArg(4, buffer_uv);
		Intersect_mesh.setArg(5, buffer_result);
		Intersect_mesh.setArg(6, cl_orig);
		Intersect_mesh.setArg(7, cl_dir);
		CL_queue.finish();
		cl::Event e;
		CL_queue.enqueueNDRangeKernel(Intersect_mesh, cl::NullRange, cl::NDRange(numTris), cl::NullRange, 0, &e);
		e.wait();
		CL_queue.enqueueReadBuffer(buffer_tNear, CL_TRUE, 0, sizeof(cl_float)*numTris, cl_tNear);
		CL_queue.enqueueReadBuffer(buffer_uv, CL_TRUE, 0, sizeof(cl_float2)*numTris, cl_uv);
		cl_int err = CL_queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, sizeof(cl_bool)*numTris, result);
		if (err < 0) {
			perror("Couldn't enqueue the kernel execution command");
			exit(1);
		}


		//system("pause");
		for (int i = 0; i < numTris; i++) {
			//std::cout << tNear << std::ends;
			//std::cout << cl_uv[i].x << " , ";
			//std::cout << cl_uv[i].y << std::endl;
			if (result[i] && cl_tNear[i] < tNear) {
				tNear = cl_tNear[i];
				uv.x = cl_uv[i].x;
				uv.y = cl_uv[i].y;
				
				delete cl_v0; delete cl_v1; delete cl_v2;
				delete cl_uv; delete result; delete cl_tNear;
				delete program;
				//system("pause");
				return true;
			}
		}
		delete cl_v0; delete cl_v1; delete cl_v2;
		delete cl_uv; delete result; delete cl_tNear;
		delete program;
		if (numTris > 500) {
			printf("%0.5f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
			system("pause");
		}

		return false;
		*/
	}
	void getSurfaceProperties(
		const Vec3f &hitPoint,
		const Vec3f &viewDirection,
		const uint32_t &triIndex,
		const Vec2f &uv,
		Vec3f &hitNormal,
		Vec2f &hitTextureCoordinates) const
	{
		if (smoothShading) {
			// vertex normal
			const Vec3f &n0 = N[triIndex * 3];
			const Vec3f &n1 = N[triIndex * 3 + 1];
			const Vec3f &n2 = N[triIndex * 3 + 2];
			hitNormal = (1 - uv.x - uv.y) * n0 + uv.x * n1 + uv.y * n2;
		}
		else {
			// face normal
			const Vec3f &v0 = P[trisIndex[triIndex * 3]];
			const Vec3f &v1 = P[trisIndex[triIndex * 3 + 1]];
			const Vec3f &v2 = P[trisIndex[triIndex * 3 + 2]];
			hitNormal = (v1 - v0).crossProduct(v2 - v0);
		}

		// doesn't need to be normalized as the N's are normalized but just for safety
		hitNormal.normalize();

		// texture coordinates
		const Vec2f &st0 = sts[triIndex * 3];
		const Vec2f &st1 = sts[triIndex * 3 + 1];
		const Vec2f &st2 = sts[triIndex * 3 + 2];
		hitTextureCoordinates = (1 - uv.x - uv.y) * st0 + uv.x * st1 + uv.y * st2;
	}
	// member variables
	uint32_t numTris;                       // number of triangles
	std::unique_ptr<Vec3f[]> P;            // triangles vertex position
	std::unique_ptr<uint32_t[]> trisIndex; // vertex index array
	std::unique_ptr<Vec3f[]> N;            // triangles vertex normals
	std::unique_ptr<Vec2f[]> sts;          // triangles texture coordinates
	bool smoothShading = true;              // smooth shading by default
	
};
TriangleMesh* loadPolyMeshFromObj(const std::string &path, const Matrix44f &o2w) {
	
	Assimp::Importer importer;
	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
	importer.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, true);
	unsigned flags =
		aiProcess_Triangulate |
		aiProcess_FlipUVs |
		aiProcess_GenNormals;

	

	const aiScene* scene = importer.ReadFile(path, flags);
	if (!scene) return false;

	const aiMesh* mesh = scene->mMeshes[0];

	uint32_t numFaces = 0;
	std::unique_ptr<uint32_t[]> faceIndex(new uint32_t[mesh->mNumFaces]);
	std::unique_ptr<uint32_t[]> vertsIndex(new uint32_t[3*mesh->mNumFaces]);
	std::unique_ptr<Vec3f[]> verts(new Vec3f[mesh->mNumVertices]);
	std::unique_ptr<Vec3f[]> normals(new Vec3f[mesh->mNumVertices]);
	std::unique_ptr<Vec2f[]> st(new Vec2f[mesh->mNumVertices]);

	numFaces = mesh->mNumFaces;

	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		faceIndex[i] = mesh->mFaces[i].mNumIndices;
	}

	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {

		vertsIndex[3*i] = mesh->mFaces[i].mIndices[0];
		vertsIndex[3*i + 1] = mesh->mFaces[i].mIndices[1];
		vertsIndex[3*i + 2] = mesh->mFaces[i].mIndices[2];
		
	}

	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
		
		aiVector3D tmp = mesh->mVertices[vertsIndex[i]];
		verts[i] = Vec3f(tmp.x, tmp.y, tmp.z);
	}

	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
		aiVector3D tmp = mesh->mNormals[i];
		normals[i] = Vec3f(tmp.x, tmp.y, tmp.z);
	}

	for (unsigned int i = 0; i < mesh->mNumAnimMeshes; i++) {
		aiVector3D UVW = mesh->mTextureCoords[0][i];
		st[i] = Vec2f(UVW.x, UVW.y);
	}

	return new TriangleMesh(o2w, numFaces, faceIndex, vertsIndex, verts, normals, st);
}

TriangleMesh* loadPolyMeshFromFile(const char *file, const Matrix44f &o2w)
{
	
	std::ifstream ifs;
	try {
		ifs.open(file);
		if (ifs.fail()) throw;
		std::stringstream ss;
		ss << ifs.rdbuf();
		uint32_t numFaces;
		ss >> numFaces;
		std::unique_ptr<uint32_t[]> faceIndex(new uint32_t[numFaces]);
		uint32_t vertsIndexArraySize = 0;
		// reading face index array
		for (uint32_t i = 0; i < numFaces; ++i) {
			ss >> faceIndex[i];
			vertsIndexArraySize += faceIndex[i];
		}
		std::unique_ptr<uint32_t[]> vertsIndex(new uint32_t[vertsIndexArraySize]);
		uint32_t vertsArraySize = 0;
		// reading vertex index array
		for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
			ss >> vertsIndex[i];
			if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
		}
		vertsArraySize += 1;
		// reading vertices
		std::unique_ptr<Vec3f[]> verts(new Vec3f[vertsArraySize]);
		for (uint32_t i = 0; i < vertsArraySize; ++i) {
			ss >> verts[i].x >> verts[i].y >> verts[i].z;
		}
		// reading normals
		std::unique_ptr<Vec3f[]> normals(new Vec3f[vertsIndexArraySize]);
		for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
			ss >> normals[i].x >> normals[i].y >> normals[i].z;
		}
		// reading st coordinates
		std::unique_ptr<Vec2f[]> st(new Vec2f[vertsIndexArraySize]);
		for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
			ss >> st[i].x >> st[i].y;
		}

		return new TriangleMesh(o2w, numFaces, faceIndex, vertsIndex, verts, normals, st);
	}
	catch (...) {
		ifs.close();
	}
	ifs.close();

	return nullptr;
}

// [comment]
// Light base class
// [/comment]
class Light
{
public:
	Light(const Matrix44f &l2w, const Vec3f &c = 1, const float &i = 1) : lightToWorld(l2w), color(c), intensity(i) {}
	virtual ~Light() {}
	virtual void illuminate(const Vec3f &P, Vec3f &, Vec3f &, float &) const = 0;
	Vec3f color;
	float intensity;
	Matrix44f lightToWorld;
};

// [comment]
// Distant light
// [/comment]
class DistantLight : public Light
{
	Vec3f dir;
public:
	DistantLight(const Matrix44f &l2w, const Vec3f &c = 1, const float &i = 1) : Light(l2w, c, i)
	{
		l2w.multDirMatrix(Vec3f(0, 0, -1), dir);
		dir.normalize(); // in case the matrix scales the light
	}
	void illuminate(const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance) const
	{
		lightDir = dir;
		lightIntensity = color * intensity;
		distance = kInfinity;
	}
};

// [comment]
// Point light
// [/comment]
class PointLight : public Light
{
	Vec3f pos;
public:
	PointLight(const Matrix44f &l2w, const Vec3f &c = 1, const float &i = 1) : Light(l2w, c, i)
	{
		l2w.multVecMatrix(Vec3f(0), pos);
	}
	// P: is the shaded point
	void illuminate(const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance) const
	{
		lightDir = (P - pos);
		float r2 = lightDir.norm();
		distance = sqrt(r2);
		lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
		// avoid division by 0
		lightIntensity = color * intensity / (4 * M_PI * distance);
	}
};

enum RayType { kPrimaryRay, kShadowRay };

struct IsectInfo
{
	const Object *hitObject = nullptr;
	float tNear = kInfinity;
	Vec2f uv;
	uint32_t index = 0;
};

bool trace(
	const Vec3f &orig, const Vec3f &dir,
	const std::vector<std::unique_ptr<Object>> &objects,
	IsectInfo &isect,
	RayType rayType = kPrimaryRay)
{
	int howManyCount = 0;
	isect.hitObject = nullptr;
	for (uint32_t k = 0; k < objects.size(); ++k) {
		float tNear = kInfinity;
		uint32_t index = 0;
		Vec2f uv;
		
		if (objects[k]->intersect(orig, dir, tNear, index, uv) && tNear < isect.tNear) {
			isect.hitObject = objects[k].get();
			isect.tNear = tNear;
			isect.index = index;
			isect.uv = uv;
		}
		howManyCount++;
	}
	return (isect.hitObject != nullptr);
}


void createCoordinateSystem(const Vec3f &N, Vec3f &Nt, Vec3f &Nb)
{
	if (std::fabs(N.x) > std::fabs(N.y))
		Nt = Vec3f(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
	else
		Nt = Vec3f(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
	Nb = N.crossProduct(Nt);
}

Vec3f uniformSampleHemisphere(const float &r1, const float &r2)
{
	// cos(theta) = u1 = y
	// cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	float sinTheta = sqrtf(1 - r1 * r1);
	float phi = 2 * M_PI * r2;
	float x = sinTheta * cosf(phi);
	float z = sinTheta * sinf(phi);
	return Vec3f(x, r1, z);
}

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0, 1);

Vec3f castRay(
	const Vec3f &orig, const Vec3f &dir,
	const std::vector<std::unique_ptr<Object>> &objects,
	const std::vector<std::unique_ptr<Light>> &lights,
	const Options &options,
	const uint32_t & depth = 0)

{
	if (depth > options.maxDepth) return 0;//options.backgroundColor;
	Vec3f hitColor = 0;
	IsectInfo isect;
	if (trace(orig, dir, objects, isect)) {
		// [comment]
		// Evaluate surface properties (P, N, texture coordinates, etc.)
		// [/comment]
		Vec3f hitPoint = orig + dir * isect.tNear;
		Vec3f hitNormal;
		Vec2f hitTexCoordinates;
		isect.hitObject->getSurfaceProperties(hitPoint, dir, isect.index, isect.uv, hitNormal, hitTexCoordinates);
		switch (isect.hitObject->type) {
			// [comment]
			// Simulate diffuse object
			// [/comment]
		case kDiffuse:
		{
			// [comment]
			// Compute direct ligthing
			// [/comment]
			Vec3f directLighting = 0;

			for (uint32_t i = 0; i < lights.size(); ++i) {
				Vec3f lightDir, lightIntensity;
				IsectInfo isectShad;
				lights[i]->illuminate(hitPoint, lightDir, lightIntensity, isectShad.tNear);
				bool vis = !trace(hitPoint + hitNormal * options.bias, -lightDir, objects, isectShad, kShadowRay);
				directLighting = vis * lightIntensity * std::max(0.f, hitNormal.dotProduct(-lightDir));
			}

			// [comment]
			// Compute indirect ligthing
			// [/comment]
			Vec3f indirectLigthing = 0;
#ifdef GI
			uint32_t N = 3;// / (depth + 1);
			Vec3f Nt, Nb;
			createCoordinateSystem(hitNormal, Nt, Nb);
			float pdf = 1 / (2 * M_PI);
#pragma omp parallel for
			for (int n = 0; n < N; ++n) {
				float r1 = distribution(generator);
				float r2 = distribution(generator);
				Vec3f sample = uniformSampleHemisphere(r1, r2);
				Vec3f sampleWorld(
					sample.x * Nb.x + sample.y * hitNormal.x + sample.z * Nt.x,
					sample.x * Nb.y + sample.y * hitNormal.y + sample.z * Nt.y,
					sample.x * Nb.z + sample.y * hitNormal.z + sample.z * Nt.z);
				// don't forget to divide by PDF and multiply by cos(theta)
				indirectLigthing += r1 * castRay(hitPoint + sampleWorld * options.bias,
					sampleWorld, objects, lights, options, depth + 1) / pdf;
			}
			// divide by N
			indirectLigthing /= (float)N;
#endif

			hitColor = (directLighting / M_PI + indirectLigthing) * isect.hitObject->albedo;
			break;
		}
		default:
			break;
		}
	}
	else {
		hitColor = 1;
	}

	return hitColor;
}
Options options;
std::vector<std::unique_ptr<Object>> objectss;
std::vector<std::unique_ptr<Light>> lightss;


void Display2x2(uint32_t i, uint32_t j, std::unique_ptr<Vec3f[]> &framebuffer, float gamma) {
	FrameBuffer::SetPixel(i, j, (255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].x, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].y, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].z, 1 / gamma))));
	++i;
	FrameBuffer::SetPixel(i, j, (255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].x, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].y, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].z, 1 / gamma))));
	--i;  ++j;
	FrameBuffer::SetPixel(i, j, (255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].x, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].y, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].z, 1 / gamma))));
	++i;
	FrameBuffer::SetPixel(i, j, (255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].x, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].y, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].z, 1 / gamma))));

}
void Display1x1(uint32_t i, uint32_t j, std::unique_ptr<Vec3f[]> &framebuffer, float gamma) {
	FrameBuffer::SetPixel(i, j, (255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].x, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].y, 1 / gamma))),
		(255 * clamp(0, 1, powf(framebuffer[WIDTH*j + i].z, 1 / gamma))));
}

void render() {

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	std::unique_ptr<Vec3f[]> framebuffer(new Vec3f[options.width * options.height]);
	Vec3f *pix = framebuffer.get();
	float scale = tan(deg2rad(options.fov * 0.5));
	float imageAspectRatio = options.width / (float)options.height;
	Vec3f orig;
	options.cameraToWorld.multVecMatrix(Vec3f(0), orig);
	auto timeStart = std::chrono::high_resolution_clock::now();
	for (uint32_t j = 0; j < options.height; ++j) {
		for (uint32_t i = 0; i < options.width; ++i) {
			// generate primary ray direction
			float x = (2 * (i + 0.5) / (float)options.width - 1) * imageAspectRatio * scale;
			float y = (1 - 2 * (j + 0.5) / (float)options.height) * scale;
			Vec3f dir;
			options.cameraToWorld.multDirMatrix(Vec3f(x, y, -1), dir);
			dir.normalize();
			*(pix++) = castRay(orig, dir, objectss, lightss, options);
			float gamma = 1;
			Display1x1(i, j, framebuffer, gamma);
		}
		glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, FrameBuffer::buffer);
		glutSwapBuffers();
		fprintf(stderr, "\r%3d%c", uint32_t(j / (float)options.height * 100), '%');
	}

	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
	fprintf(stderr, "\rDone: %.2f (sec)\n", passedTime / 1000);
}

class ColorDiff {
public:
	Vec2f pos;
	uint32_t colorDiff;

	ColorDiff(Vec2f pos1, Vec3f color1, Vec2f pos2, Vec3f color2) {
		if (color1.x + color1.y + color1.z > color2.x + color2.y + color2.z) {
			pos = pos1;
			colorDiff = color1.x + color1.y + color1.z;
		}
		else {
			pos = pos2;
			colorDiff = color2.x + color2.y + color2.z;
		}
	}
};

priority_queue<ColorDiff, vector<ColorDiff>, less<uint32_t>> anti_aliasing;
void render_ver_location()
{
	std::unique_ptr<Vec3f[]> framebuffer(new Vec3f[options.width * options.height]);
	Vec3f *pix = framebuffer.get();
	float scale = tan(deg2rad(options.fov * 0.5));
	float imageAspectRatio = options.width / (float)options.height;
	Vec3f orig;
	options.cameraToWorld.multVecMatrix(Vec3f(0), orig);
	auto timeStart = std::chrono::high_resolution_clock::now();

	for (uint32_t j = 0; j < options.height; j += 2) {
		for (uint32_t i = 0; i < options.width; i += 2) {
			float x = (2 * (i + 0.5) / (float)options.width - 1) * imageAspectRatio * scale;
			float y = (1 - 2 * (j + 0.5) / (float)options.height) * scale;
			Vec3f dir;
			options.cameraToWorld.multDirMatrix(Vec3f(x, y, -1), dir);
			dir.normalize();
			Vec3f result = castRay(orig, dir, objectss, lightss, options);
			*pix = result;
			*(pix + WIDTH) = result;
			*(++pix) = result;
			*(pix + WIDTH) = result;
			*(pix++);
			float gamma = 1;
			Display2x2(i, j, framebuffer, gamma);

		}
		glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, FrameBuffer::buffer);
		glutSwapBuffers();
		pix = pix + WIDTH;
		fprintf(stderr, "\r%3d%c", uint32_t(j / (float)options.height * 100), '%');
	}

	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
	fprintf(stderr, "\rDone: %.2f (sec)\n", passedTime / 1000);
	
}

void Init_CL() {
	//stl vector to store all of the available platforms
	//get all available platforms

	std::cout << platforms.size() << std::endl;
	cl::Platform::get(&platforms);

	if (platforms.size() == 0)
	{
		std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
		exit(1);
	}

	//Create a stl vector to store all of the availbe devices to use from the first platform.
	//Get the available devices from the platform. For me the platform for my 980ti is actually th e second in the platform list but for simplicity we will use the first one.
	platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//Set the device to the first device in the platform. You can have more than one device associated with a single platform, for instance if you had two of the same GPUs on your system in SLI or CrossFire.
	device = devices[0];

	//This is just helpful to see what device and platform you are using.
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;


	//Finally create the OpenCL context from the device you have chosen.
	kernel_code = new std::string(readFile("simple_add.cl"));
	sources = new cl::Program::Sources();
	sources->push_back({ kernel_code->c_str(),kernel_code->length() });

	context = new cl::Context(device);
}

void init(void)
{
	//Init_CL();
	FrameBuffer::Init(WIDTH, HEIGHT);
	//Initialize everything here
	// loading gemetry
	// lights

	// aliasing example
	options.fov = 80;
	options.width = WIDTH;
	options.height = HEIGHT;
	options.cameraToWorld = //Matrix44f(0.965926, 0, -0.258819, 0, 0.0066019, 0.999675, 0.0246386, 0, 0.258735, -0.0255078, 0.965612, 0, 0.764985, 0.791882, 5.868275, 1);
		Matrix44f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 5, 1);

	TriangleMesh *cube = loadPolyMeshFromFile("./cubegi.geo", Matrix44f::kIdentity);
	if (cube != nullptr) {
		cube->albedo = Vec3f(0.188559, 0.287, 0.200726);
		objectss.push_back(std::unique_ptr<Object>(cube));
	}
	TriangleMesh *plane = loadPolyMeshFromFile("./planegi.geo", Matrix44f::kIdentity);
	if (plane != nullptr) {
		plane->albedo = Vec3f(0.2, 0.2, 0.15);
		objectss.push_back(std::unique_ptr<Object>(plane));
	}
	TriangleMesh *plane2 = loadPolyMeshFromFile("./planegi2.geo", Matrix44f::kIdentity);
	if (plane2 != nullptr) {
		plane2->albedo = Vec3f(0, 0, 0.18);
		objectss.push_back(std::unique_ptr<Object>(plane2));
	}
	TriangleMesh *plane3 = loadPolyMeshFromFile("./planegi3.geo", Matrix44f::kIdentity);
	if (plane3 != nullptr) {
		plane3->albedo = Vec3f(0.2, 0.2, 0.15);
		objectss.push_back(std::unique_ptr<Object>(plane3));
	}
	TriangleMesh *plane4 = loadPolyMeshFromFile("./planegi4.geo", Matrix44f::kIdentity);
	if (plane4 != nullptr) {
		plane4->albedo = Vec3f(0.18, 0, 0);
		objectss.push_back(std::unique_ptr<Object>(plane4));
	}
	TriangleMesh *plane5 = loadPolyMeshFromFile("./plangegi5.geo", Matrix44f::kIdentity);
	if (plane5 != nullptr) {
		plane5->albedo = Vec3f(0.2, 0.2, 0.15);
		objectss.push_back(std::unique_ptr<Object>(plane5));
	}
	TriangleMesh *test = loadPolyMeshFromObj("./teapot.obj", Matrix44f::kIdentity);
	if (test != nullptr) {
		test->albedo = Vec3f(0.2, 0.2, 0.15);
		objectss.push_back(std::unique_ptr<Object>(test));
	}


	Matrix44f xformSphere;

	xformSphere[3][1] = 1;
	Sphere *sph = new Sphere(xformSphere, 1);
	objectss.push_back(std::unique_ptr<Object>(sph));

	//Matrix44f l2w(0.916445, -0.218118, 0.335488, 0, 0.204618, -0.465058, -0.861309, 0, 0.343889, 0.857989, -0.381569, 0, 0, 0, 0, 1);
	//Matrix44f l2w2(1, 0, 0,0,   0, 1, 0, 0,    0, 0, 1 ,0   , 0, -2, 5, 1);
	Matrix44f l2w3(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 3, 0, 1);

	//lightss.push_back(std::unique_ptr<Light>(new DistantLight(l2w, 1, 500)));
	//lightss.push_back(std::unique_ptr<Light>(new PointLight(l2w2, 1, 1500)));
	lightss.push_back(std::unique_ptr<Light>(new PointLight(l2w3, 1, 1500)));

}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case VK_ESCAPE:
		delete[] FrameBuffer::buffer;
		glutDestroyWindow(winID);
		exit(0);
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	switch (button)
	{
		Sleep(1000);
	}
}

void mouseMove(int x, int y)
{
	switch (x&&y)
	{
		return;
	}
}

void loop(void)
{
	glutPostRedisplay();
}


int main(int argc, char **argv)
{

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitWindowPosition(500, 100);

	winID = glutCreateWindow("CS200");

	glClearColor(0, 0, 0, 1);


	glutKeyboardFunc(keyboard);
	glutDisplayFunc(render);
//	glutDisplayFunc(render_ver_location);
	glutIdleFunc(loop);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseMove);

	init();

	glutMainLoop();


	return 0;
}
