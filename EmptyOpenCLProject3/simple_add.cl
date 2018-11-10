#ifdef GPU_KERNEL
printf("where is");
#endif

#define swap(a,b) {float temp; temp = a; a = b; b = temp; }

typedef struct _tri {
	float3 v0;
	float3 v1;
	float3 v2;
}tri;

typedef struct _MeshTriangle{
	float3 *P;
	int *trisIndex;
	float3 *N;
	float2 *sts;
}MeshTriangle;


bool rayTriangleIntersect(float3 orig,float3 dir, float3 v0,
	float3 v1, float3 v2, float *t, float *u, float *v) {
	


	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;
	float3 pvec = cross(dir, v0v2);
	float det = dot(v0v1, pvec);
	
	if (fabs(det) < 1e-8) return false;

	float invDet = 1 / det;
	
	float3 tvec = orig - v0;
	*u = dot(tvec, pvec) * invDet;
	if (*u < 0 || *u > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	*v = dot(dir, qvec) * invDet;
	if (*v < 0 || *u + *v > 1) return false;

	*t = dot(v0v2,qvec) * invDet;

	return (*t > 0) ? true : false;
}

bool Intersect(tri triangles, const float3 orig, const float3 dir, global float *tNear,global float2 *uv) {
	bool intersect = false;
	float t; float u; float v;
	if (rayTriangleIntersect(orig, dir,triangles.v0, triangles.v1, triangles.v2, &t, &u, &v) && t < tNear[get_global_id(0)]) {
		tNear[get_global_id(0)] = t;
		uv[get_global_id(0)].x = u;
		uv[get_global_id(0)].y = v;
		intersect |= true;
	}



	return intersect;
}
//
//bool Intersect_Sphere(float r,float3 orig, float3 dir, float tNear){
//	float radius2 = r * r;
//
//	float t0, t1;
//	
//	float3 L = orig - center;
//	float a = dot(dir, dir);
//	float b = 2 * dot(dir, L);
//	float c = dot(L, L) - radius2;
//	if (!solveQuadratic(a, b, c, t0, t1)) return false;
//
//	if (t0 > t1) swap(t0, t1);
//
//	if(t0 < 0){ 
//		t0 = t1;
//		if (t0 < 0) return false;
//	}
//	tNear = t0;
//
//	return true;
//}



kernel void Intersect_mesh(global float3 *v0, global float3 *v1, global float3 *v2, global float *tNear, global float2 *uv, global bool *result, float3 orig, float3 dir) {
	int gid = get_global_id(0);
	tri triangles;
	triangles.v0 = v0[get_global_id(0)];
	triangles.v1 = v1[get_global_id(0)];
	triangles.v2 = v2[get_global_id(0)];
		
	result[get_global_id(0)] = Intersect(triangles, orig, dir, tNear, uv);

}