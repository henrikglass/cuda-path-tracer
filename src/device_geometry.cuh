
__device__ bool intersect_triangle(
        vec3 v0, 
        vec3 v1, 
        vec3 v2, 
        Triangle *triangle,
        Entity *entity, 
        Intersection &bestHit, 
        const Ray &ray
);

__device__ bool intersects_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const Ray &ray,
        const Intersection &bestHit
);

__device__
inline bool Octree::get_closest_intersection(
        Vertex *vertices, 
        Triangle *triangles, 
        const Ray &ray, 
        Intersection &bestHit,
        Entity *entity
) {
    if(!this->region.intersects(ray, bestHit))
        return false;
    
    // check intersections for triangle in this node
    bool hit = false;
    for (int i = 0; i < this->n_triangle_indices; i++) {
        int tri_idx = this->d_triangle_indices[i];
        Triangle *triangle = &(triangles[tri_idx]);
        vec3 v0 = vertices[triangle->idx_a].position;
        vec3 v1 = vertices[triangle->idx_b].position;
        vec3 v2 = vertices[triangle->idx_c].position;
        hit = intersect_triangle(v0, v1, v2, triangle, entity, bestHit, ray) || hit;
    }

    // Check children
    for (int i = 0; i < 8; i++) {
        if(this->d_children[i] != nullptr) {
            hit = d_children[i]->get_closest_intersection(vertices, triangles, ray, bestHit, entity) || hit;
        }
    }

    return hit;
}

/*
 * Revelles et. al. parametric octree traversal algorithm.
 * Modified from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.3092&rep=rep1&type=pdf
 */
__device__ 
inline bool Octree::ray_step(
        const Ray &ray, 
        Intersection &bestHit,
        Entity *entity
) {
    Ray r(ray);
    vec3 oct_mid = 0.5f * (this->region.max + this->region.min);

    unsigned char a = 0;
    if (r.direction.x < 0.0f ) {
        float delta = r.origin.x - oct_mid.x;
        r.origin.x = oct_mid.x - delta;
        r.direction.x = -(r.direction.x);
        a |= 0b100; // 4
    }
    if (r.direction.y < 0.0f ) {
        float delta = r.origin.y - oct_mid.y;
        r.origin.y = oct_mid.y - delta;
        r.direction.y = -(r.direction.y);
        a |= 0b010; // 2
    }
    if (r.direction.z < 0.0f) {
        float delta = r.origin.z - oct_mid.z;
        r.origin.z = oct_mid.z - delta;
        r.direction.z = -(r.direction.z);
        a |= 0b001; // 1
    }

    if (a != 0) 
        r.recalc_fracs();

    float tx0 = (this->region.min.x - r.origin.x) * r.fracs.x;
    float tx1 = (this->region.max.x - r.origin.x) * r.fracs.x;
    float ty0 = (this->region.min.y - r.origin.y) * r.fracs.y;
    float ty1 = (this->region.max.y - r.origin.y) * r.fracs.y;
    float tz0 = (this->region.min.z - r.origin.z) * r.fracs.z;
    float tz1 = (this->region.max.z - r.origin.z) * r.fracs.z;
    float tmin = fmaxf(fmaxf(tx0, ty0), tz0);
    float tmax = fminf(fminf(tx1, ty1), tz1);

    if ((tmin < tmax) && (tmax > 0.0f)) {
        return proc_subtree(
            a,
            vec3(tx0, ty0, tz0),
            vec3(tx1, ty1, tz1),
            ray, bestHit, entity
        );
    }

    return false;
}

__device__
inline unsigned char find_first_node(const vec3 &t0, const vec3 &tM) {
    unsigned char answer = 0;
    if (t0.x > t0.y) {
        if (t0.x > t0.z) { // YZ plane
            if (tM.y < t0.x) answer |= 2;//0b010;
            if (tM.z < t0.x) answer |= 1;//0b001;
            return answer;
        }  
    } else if (t0.y > t0.z) { // XZ plane
        if (tM.x < t0.y) answer |= 4;//0b100;
        if (tM.z < t0.y) answer |= 1;//0b001;
        return answer;
    } 
    // XY plane
    if (tM.x < t0.z) answer |= 4;//0b100;
    if (tM.y < t0.z) answer |= 2;//0b010;
    return answer;
}

__device__
inline unsigned char next_node(
        float txm, float tym, float tzm, 
        unsigned char x, unsigned char y, unsigned char z
) {
    if (txm < tym) {
        if (txm < tzm) return x; // YZ plane
    } else {
        if (tym < tzm) return y;
    }
    return z;
}

__device__ 
inline bool Octree::proc_subtree(
        unsigned char a,
        vec3 t0, 
        vec3 t1,
        const Ray &ray, 
        Intersection &bestHit,
        Entity *entity
) {
    unsigned char curr_node;

    if ((t1.x < 0.0f) || (t1.y < 0.0f) || (t1.y < 0.0f))
        return false;

    // if leaf check intersections for triangle in this node
    if (this->depth == MAX_OCTREE_DEPTH) {
        bool hit = false;
        for (int i = 0; i < this->n_triangle_indices; i++) {
            int tri_idx = this->d_triangle_indices[i];
            Triangle *triangle = &(entity->d_triangles[tri_idx]);
            vec3 v0 = entity->d_vertices[triangle->idx_a].position;
            vec3 v1 = entity->d_vertices[triangle->idx_b].position;
            vec3 v2 = entity->d_vertices[triangle->idx_c].position;
            hit = intersect_triangle(v0, v1, v2, triangle, entity, bestHit, ray) || hit;
        }
        return hit;
    }

    // determine center
    vec3 tM = 0.5f * (t0 + t1);

    curr_node = find_first_node(t0, tM);
    bool hit = false;
    do {
        unsigned char adj_curr_node = curr_node ^ a;
        if (this->d_children[adj_curr_node] != nullptr) {
            switch (curr_node) {
                case 0: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(t0.x, t0.y, t0.z), vec3(tM.x, tM.y, tM.z), ray, bestHit, entity) || hit; break;
                case 1: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(t0.x, t0.y, tM.z), vec3(tM.x, tM.y, t1.z), ray, bestHit, entity) || hit; break;
                case 2: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(t0.x, tM.y, t0.z), vec3(tM.x, t1.y, tM.z), ray, bestHit, entity) || hit; break;
                case 3: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(t0.x, tM.y, tM.z), vec3(tM.x, t1.y, t1.z), ray, bestHit, entity) || hit; break;
                case 4: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(tM.x, t0.y, t0.z), vec3(t1.x, tM.y, tM.z), ray, bestHit, entity) || hit; break;
                case 5: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(tM.x, t0.y, tM.z), vec3(t1.x, tM.y, t1.z), ray, bestHit, entity) || hit; break;
                case 6: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(tM.x, tM.y, t0.z), vec3(t1.x, t1.y, tM.z), ray, bestHit, entity) || hit; break;
                case 7: hit = this->d_children[adj_curr_node]->proc_subtree(a, vec3(tM.x, tM.y, tM.z), vec3(t1.x, t1.y, t1.z), ray, bestHit, entity) || hit; break;
            }
        }
        switch (curr_node) {
            case 0: curr_node = next_node(tM.x, tM.y, tM.z, 4, 2, 1); break;
            case 1: curr_node = next_node(tM.x, tM.y, t1.z, 5, 3, 8); break;
            case 2: curr_node = next_node(tM.x, t1.y, tM.z, 6, 8, 3); break;
            case 3: curr_node = next_node(tM.x, t1.y, t1.z, 7, 8, 8); break;
            case 4: curr_node = next_node(t1.x, tM.y, tM.z, 8, 6, 5); break;
            case 5: curr_node = next_node(t1.x, tM.y, t1.z, 8, 7, 8); break;
            case 6: curr_node = next_node(t1.x, t1.y, tM.z, 8, 8, 7); break;
            case 7: curr_node = 8; break;
        }
    } while (curr_node < 8 && !hit);

    return hit;
}

__device__
inline bool get_closest_intersection_in_scene(const Ray &ray, Entity *entities, int n_entities, Intersection &is) {
    bool is_hit = false;
    //long t1 = clock();
    for (int i = 0; i < n_entities; i++) {
        is_hit = entities[i].get_closest_intersection(ray, is) || is_hit;
    }
    //long t2 = clock();
    //printf("t: %ld\n", t2 - t1);

    // if hit entity has smooth_shading enabled, adjust the normal
    Triangle *tr = is.triangle;
    Entity *e = is.entity;
    if (is_hit && tr != nullptr) {
        float u = is.u;
        float v = is.v;
        float w = 1.0f - (u + v);

        //printf("in: u,v:   %g, %g\n", u, v);

        // interpolate uv:s
        if (e->d_material->has_albedo_map /*or any other texture*/ ) {
            vec2 v0_uv = e->d_uvs[tr->vt_idx_a];
            //printf("v0: u,v:   %g, %g\n", v0_uv.x, v0_uv.y);
            vec2 v1_uv = e->d_uvs[tr->vt_idx_b];
            //printf("v1: u,v:   %g, %g\n", v0_uv.x, v0_uv.y);
            vec2 v2_uv = e->d_uvs[tr->vt_idx_c];
            //printf("v2: u,v:   %g, %g\n", v0_uv.x, v0_uv.y);
            vec2 intp_uv = u * v1_uv + v * v2_uv + w * v0_uv;
            is.u = intp_uv.x;
            is.v = intp_uv.y;
        }

        //printf("out: u,v:   %g, %g\n", is.u, is.v);

        // interpolate normals for smooth shading
        if (e->d_material->smooth_shading) {
            vec3 v0_normal = e->d_vertices[tr->idx_a].normal;
            vec3 v1_normal = e->d_vertices[tr->idx_b].normal;
            vec3 v2_normal = e->d_vertices[tr->idx_c].normal;
            is.normal = u * v1_normal + v * v2_normal + w * v0_normal; // pure guess
            is.normal.normalize();
        }
    }

    return is_hit;
}

__device__
inline bool Entity::get_closest_intersection(const Ray &ray, Intersection &bestHit) {
    switch (this->shape) {
        case SPHERE:
            return get_closest_sphere_intersection(ray, bestHit);
        case TRIANGLE_MESH:
            return get_closest_triangle_mesh_intersection(ray, bestHit);
        default:
            return false;
    }
}

__device__
inline bool Entity::get_closest_sphere_intersection(const Ray &ray, Intersection &bestHit) {
    vec3 d = ray.origin - this->center;
    float p1 = -dot(ray.direction, d);
    float p2sqr = __fsub_rn(__fmul_rn(p1, p1), dot(d,d)) + __fmul_rn(this->radius, this->radius);
    if (p2sqr < 0)
        return false;
    float p2 = __fdividef(1.0f, __frsqrt_rn(p2sqr)); // sqrt(p2sqr)
    float t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
    if (t > 0 && t < bestHit.distance)
    {
        bestHit.distance = t;
        bestHit.position = ray.origin + t * ray.direction;
        bestHit.normal = bestHit.position - this->center;
        bestHit.normal.normalize();
        bestHit.entity = this;
        return true;
    }
    return false;
}

__device__
inline bool Entity::get_closest_triangle_mesh_intersection(const Ray &ray, Intersection &bestHit) {
    if (!this->aabb.intersects(ray, bestHit))
        return false;
    
    // no octree. Check against all triangles.
    if (this->octree == nullptr) {
        //printf("as list");
        bool hit = false;
        for (size_t i = 0; i < this->n_triangles; i++) {
            hit = intersects_triangle(&(this->d_triangles[i]), bestHit, ray) || hit;
        }
        return hit;
    } else {
        //printf("as octree");
        //get_closest_intersection
        return this->d_octree->ray_step(
                ray, 
                bestHit, 
                this
        );
    }
}

__device__
inline bool Entity::intersects_triangle(Triangle *triangle, Intersection &bestHit, const Ray &ray) {
    vec3 v0 = this->d_vertices[triangle->idx_a].position;
    vec3 v1 = this->d_vertices[triangle->idx_b].position;
    vec3 v2 = this->d_vertices[triangle->idx_c].position;
    return intersect_triangle(v0, v1, v2, triangle, this, bestHit, ray);
}

/*
 * Tomas Akenine-Möller and Ben Trumbore's algorithm.
 *
 * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/raytri_tam.pdf
 */
__device__ 
inline bool intersect_triangle(
        vec3 v0, 
        vec3 v1, 
        vec3 v2, 
        Triangle *triangle,
        Entity *entity, 
        Intersection &bestHit, 
        const Ray &ray
) {

    vec3 e1, e2, pvec, tvec, qvec;
    float t, u, v, det, inv_det;

    e1 = v1 - v0;
    e2 = v2 - v0;

    // @Incomplete Might want to to back-face culling in the future.

    pvec = cross(ray.direction, e2);
    det = dot(e1, pvec);
    if (fabs(det) < EPSILON) 
        return false;
    
    inv_det = __fdividef(1.0f, det); // 1.0f / det
    tvec = ray.origin - v0;
    u = dot(tvec, pvec) * inv_det;
    if (u < -0.0001f || u > 1.0001f)
        return false;

    qvec = cross(tvec, e1);
    v = dot(ray.direction, qvec) * inv_det;
    if (v < -0.0001f || u + v > 1.0001f)
        return false;

    t = dot(e2, qvec) * inv_det; 

    if(t > 0 && t < bestHit.distance) {
        bestHit.distance = t;
        bestHit.position = ray.origin + t * ray.direction;
        bestHit.normal = cross(e1, e2).normalized();
        bestHit.entity = entity;
        bestHit.triangle = triangle;
        bestHit.u = u;
        bestHit.v = v;
        return true;
    }

    return false;
}

__device__ 
inline bool AABB::intersects(const Ray &ray, const Intersection &bestHit) {
    return intersects_aabb(
            this->min.x,
            this->min.y,
            this->min.z,
            this->max.x,
            this->max.y,
            this->max.z,
            ray,
            bestHit
    );
}

__device__
inline bool intersects_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const Ray &ray,
        const Intersection &bestHit
) {
    float tx1 = (min_x - ray.origin.x) * ray.fracs.x;
    float tx2 = (max_x - ray.origin.x) * ray.fracs.x;
    float ty1 = (min_y - ray.origin.y) * ray.fracs.y;
    float ty2 = (max_y - ray.origin.y) * ray.fracs.y;
    float tz1 = (min_z - ray.origin.z) * ray.fracs.z;
    float tz2 = (max_z - ray.origin.z) * ray.fracs.z;

    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
    
    // box behind
    if (tmax < 0.0f)
        return false;
    
    // no intersection
    if (tmin > tmax)
        return false;

    // we've already intersected some entity nearer ray origin 
    if (bestHit.distance < tmin)
        return false;

    return true;

}