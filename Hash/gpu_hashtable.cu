
#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <assert.h>
// #include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// 
// #include "common/error_handler.cu"
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */


using u64 = unsigned long long;
using u32 = unsigned long;

using column_type = u64;
using tuple_type = column_type*;

// TODO: use thrust vector as tuple type??
// using t_gpu_index = thrust::device_vector<u64>;
// using t_gpu_tuple = thrust::device_vector<u64>;

// using t_data_internal = thrust::device_vector<u64>;
/**
 * @brief u64* to store the actual relation tuples, for serialize concern
 * 
 */
using t_data_internal = u64*;

/**
 * @brief TODO: remove this use comparator function
 * 
 * @param t1 
 * @param t2 
 * @param l 
 * @return true 
 * @return false 
 */
 __host__
 __device__
inline bool tuple_eq(tuple_type t1, tuple_type t2, u64 l) {
    for (int i = 0; i < l; i++) {
        if (t1[i] != t2[i]) {
            return false;
        }
    }
    return true;
}

struct t_equal {
    u64 arity;

    t_equal(u64 arity) {
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {
        for (int i = 0; i < arity; i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief fnv1-a hash used in original slog backend
 * 
 * @param start_ptr 
 * @param prefix_len 
 * @return __host__ __device__
 */
__host__
__device__
inline u64 prefix_hash(tuple_type start_ptr, u64 prefix_len)
{
    const u64 base = 14695981039346656037ULL;
    const u64 prime = 1099511628211ULL;

    u64 hash = base;
    for (u64 i = 0; i < prefix_len; ++i)
    {
        u64 chunk = start_ptr[i];
        hash ^= chunk & 255ULL;
        hash *= prime;
        for (char j = 0; j < 7; ++j)
        {
            chunk = chunk >> 8;
            hash ^= chunk & 255ULL;
            hash *= prime;
        }
    }
    return hash;
}

// change to std
struct tuple_less {

    // u64 *index_columns;
    u64 index_column_size;
    int arity;
    
    tuple_less(u64 index_column_size, int arity) {
        // this->index_columns = index_columns;
        this->index_column_size = index_column_size;
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {
        // fetch the index
        // column_type indices_r[index_column_size];
        // for (u64 i = 0; i < index_column_size; i++) {
        //     indices_l[i] = lhs[index_columns[i]];
        // }
        // column_type indices_r[index_column_size];
        // for (u64 i = 0; i < index_column_size; i++) {
        //     indices_r[i] = rhs[index_columns[i]];
        // }
        // compare hash first, could be index very different but share the same hash
        if (prefix_hash(lhs, index_column_size) == prefix_hash(rhs, index_column_size)) {
            // same hash
            for (u64 i = 0; i < arity; i++) {
                if (lhs[i] < rhs[i]) {
                    return true;
                }
            }
            return false;
        } else if (prefix_hash(lhs, index_column_size) < prefix_hash(rhs, index_column_size)) {
            return true;
        } else {
            return false;
        }
    }
};

/**
 * @brief A hash table entry
 * TODO: no need for struct actually, a u64[2] should be enough, easier to init
 * 
 */
struct MEntity {
    // index position in actual index_arrary
    u64 key;
    // tuple position in actual data_arrary
    u64 value;
};

#define EMPTY_HASH_ENTRY ULLONG_MAX
/**
 * @brief a C-style hashset indexing based relation container.
 *        Actual data is still stored using sorted set.
 *        Different from normal btree relation, using hash table storing the index to accelarte
 *        range fetch.
 *        Good:
 *           - fast range fetch, in Shovon's ATC paper it shows great performance.
 *           - fast serialization, its very GPU friendly and also easier for MPI inter-rank comm
 *             transmission.
 *        Bad:
 *           - need reconstruct index very time tuple is inserted (need more reasonable algorithm).
 *           - sorting is a issue, each update need resort everything seems stupid.
 * 
 */
struct GHashRelContainer {
    MEntity* index_map;
    u64 index_map_size;
    // TODO: add load factor support
    float index_map_load_factor;

    // don't have to be u64,int is enough
    // u64 *index_columns;
    u64 index_column_size;

    tuple_type* tuples;
    // flatten tuple data
    column_type* data_raw;
    // number of tuples
    u64 tuple_counts;
    // actual tuple rows in flatten data, this maybe different from
    // tuple_counts when deduplicated
    u64 data_raw_row_size;
    int arity;
};

// kernals

/**
 * @brief fill in index hash table for a relation in parallel, assume index is correctly initialized, data has been loaded
 *        , deduplicated and sorted
 * 
 * @param target the hashtable to init
 * @return dedeuplicated_bitmap 
 */
__global__
void calculate_index_hash(GHashRelContainer* target) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->tuple_counts) return;

    u64 stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < target->tuple_counts; i += stride) {
        // opt this to memcpy?
        tuple_type cur_tuple = target->tuples[i];
        // u64 indexed_cols[target->index_column_size];
        // for (size_t idx_i = 0; idx_i < target->index_column_size; idx_i ++) {
        //     indexed_cols[idx_i] = target->tuples[(i * target->arity)][target->index_columns[idx_i]];
        // }

        u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
        u64 request_hash_index = hash_val % target->index_map_size;
        u64 position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(target->index_map[position].key), EMPTY_HASH_ENTRY, hash_val);
            // check hash collision 
            // TODO: add stat log for collision ration
            if (existing_key == EMPTY_HASH_ENTRY) {
                target->index_map[position].value = i;
                break;
            } else if(existing_key == hash_val) {
                // hash for tuple's index column has already been recorded
                break;
            }
            
            position = (position + 1) % target->index_map_size;
        }
    }
}

/**
 * @brief count how many non empty hash entry in index map
 * 
 * @param target target relation hash table
 * @param size return the size
 * @return __global__ 
 */
__global__
void count_index_entry_size(GHashRelContainer* target, u64* size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->index_map_size) return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < target->index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            atomicAdd(size, 1);
        }
    }
}

/**
 * @brief rehash to make index map more compact, the new index hash size is already update in target
 *        new index already inited to empty table and have new size.
 * 
 * @param target 
 * @param old_index_map index map before compaction
 * @param old_index_map_size original size of index map before compaction
 * @return __global__ 
 */
__global__
void shrink_index_map(GHashRelContainer* target, MEntity* old_index_map, u64 old_index_map_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= old_index_map_size) return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < old_index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            u64 hash_val = target->index_map[i].key;
            u64 position = hash_val % target->index_map_size;
            while(true) {
                u64 existing_key = atomicCAS(&target->index_map[position].key, EMPTY_HASH_ENTRY, hash_val);
                if (existing_key == EMPTY_HASH_ENTRY) {
                    target->index_map[position].key = hash_val;
                    break;
                } else if(existing_key == hash_val) {
                    // hash for tuple's index column has already been recorded
                    break;
                }
                position = (position + 1) % target->index_column_size;
            }
        }
    }
}


// NOTE: must copy size out of gpu kernal code!!!
/**
 * @brief acopy the **index** from a relation to another, please use this together with *copy_data*, and settle up all metadata before copy
 * 
 * @param source source relation
 * @param destination destination relation
 * @return __global__ 
 */
__global__
void acopy_entry(GHashRelContainer* source, GHashRelContainer* destination) {
    auto source_rows = source->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        destination->index_map[i].key = source->index_map[i].key;
        destination->index_map[i].value = source->index_map[i].value;
    }
}
__global__
void acopy_data(GHashRelContainer *source, GHashRelContainer *destination) {
    auto data_rows = source->tuple_counts;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= data_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < data_rows; i += stride) {
        tuple_type cur_src_tuple = source->tuples[i]; 
        for (int j = 0; j < source->arity; j++) {
            destination->data_raw[i*source->arity+j] = cur_src_tuple[j];
        }
        destination->tuples[i] = destination->tuples[i*source->arity];
    }
}

// 
/**
 * @brief a CUDA kernel init the index entry map of a hashtabl
 * 
 * @param target the hashtable to init
 * @return void 
 */
__global__
void init_index_map(GHashRelContainer* target) {
    auto source = target->index_map;
    auto source_rows = target->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < source_rows; i += stride) {
        source[i].key = EMPTY_HASH_ENTRY;
        source[i].value = EMPTY_HASH_ENTRY;
    }
}

// a helper function to init an unsorted tuple arrary from raw data
__global__
void init_tuples_unsorted(tuple_type* tuples, column_type* raw_data, int arity, u64 rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows) return;

    int stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < rows; i += stride) {
        tuples[i] = raw_data + i * arity;
    }
}

/**
 * @brief for all tuples in outer table, match same prefix with inner table
 *
 * @note can we use pipeline here? since many matching may acually missing
 * 
 * @param inner_table the hashtable to iterate
 * @param outer_table the hashtable to match
 * @param join_column_counts number of join columns (inner and outer must agree on this)
 * @param  return value stored here, size of joined tuples
 * @return void 
 */
__global__
void get_join_result_size(GHashRelContainer* inner_table,
                          GHashRelContainer* outer_table,
                          int join_column_counts,
                          u64* join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts) return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        u64 tuple_raw_pos = i*((u64)outer_table->arity);
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols, outer_table->index_column_size * sizeof(column_type));
        // for (size_t idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i * outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        u64 current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table 
        u64 index_position = hash_val % inner_table->index_map_size;
        // 64 bit hash is less likely to have collision
        // partially solve hash conflict? maybe better for performance
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key == EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }
        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
        while (true) {
            bool cmp_res = tuple_eq(inner_table->tuples[position],
                                    outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                current_size++;
            }else {
                tuple_type cur_inner_tuple = inner_table->tuples[position];
                u64 inner_tuple_hash = prefix_hash(cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
            }
            position = position + 1;
            if (position > (inner_table->tuple_counts - 1)) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
        // cudaFree(outer_indexed_cols);
    }
}

/**
 * @brief compute the join result
 * 
 * @param inner_table 
 * @param outer_table 
 * @param join_column_counts 
 * @param output_reorder_array reorder array for output relation column selection, arrary pos < inner->arity is index in inner, > is index in outer.
 * @param output_arity output relation arity
 * @param output_raw_data join result, need precompute the size
 * @return __global__ 
 */
__global__
void get_join_result(GHashRelContainer* inner_table,
                     GHashRelContainer* outer_table,
                     int join_column_counts,
                     int* output_reorder_array,
                     int output_arity,
                     column_type* output_raw_data,
                     u64* res_count_array,
                     u64* res_offset) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts) return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        u64 tuple_raw_pos = i*((u64)outer_table->arity);
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols, outer_table->index_column_size * sizeof(column_type));
        // for (size_t idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i * outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table 
        u64 index_position = hash_val % inner_table->index_map_size;
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key == EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }

        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
        while (true) {
            // TODO: always put join columns ahead? could be various benefits but memory is issue to mantain multiple copies
            bool cmp_res = tuple_eq(inner_table->tuples[position],
                                    outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // tuple prefix match, join here
                tuple_type inner_tuple = inner_table->tuples[position];
                tuple_type new_tuple = output_raw_data + (res_offset[i] + current_new_tuple_cnt) * output_arity;

                for (int j = 0; j < output_arity; j++) {
                    if (output_reorder_array[j] < inner_table->arity) {
                        new_tuple[j] = inner_tuple[output_reorder_array[j]];
                    } else {
                        new_tuple[j] = outer_tuple[output_reorder_array[j] - inner_table->arity];
                    }
                }
                current_new_tuple_cnt++;
                if (current_new_tuple_cnt >= res_count_array[i]) {
                    return;
                }
            }else {
                // if not prefix not match, there might be hash collision
                tuple_type cur_inner_tuple = inner_table->tuples[position];
                u64 inner_tuple_hash = prefix_hash(cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
            }
            position = position + 1;
            if (position > (inner_table->tuple_counts - 1)) {
                // end of data arrary
                break;
            }
        }
    }
}

///////////////////////////////////////////////////////
// test helper

void print_hashes(GHashRelContainer* target, const char *rel_name) {
    MEntity* host_map;
    cudaMallocHost((void**) &host_map, target->index_map_size * sizeof(MEntity));
    cudaMemcpy(host_map, target->index_map, target->index_map_size * sizeof(MEntity),
               cudaMemcpyDeviceToHost);
    std::cout << "Relation hash >>> " << rel_name << std::endl;
    for (u64 i = 0; i < target->index_map_size; i++) {
        std::cout << host_map[i].key << "    " << host_map[i].value << std::endl;
    }
    std::cout << "end <<<" << std::endl;
    cudaFreeHost(host_map);
}

void print_tuple_rows(GHashRelContainer* target, const char *rel_name) {
    tuple_type* tuples_host;
    cudaMallocHost((void**) &tuples_host, target->tuple_counts * sizeof(tuple_type));
    cudaMemcpy(tuples_host, target->tuples, target->tuple_counts * sizeof(tuple_type),
               cudaMemcpyDeviceToHost);
    std::cout << "Relation tuples >>> " << rel_name << std::endl;
    std::cout << "Total tuples counts:  " <<  target->tuple_counts << std::endl;
    for (u64 i = 0; i < target->tuple_counts; i++) {
        tuple_type cur_tuple = tuples_host[i];
        tuple_type cur_tuple_host;
        cudaMallocHost((void**) &cur_tuple_host, target->arity * sizeof(column_type));
        cudaMemcpy(cur_tuple_host, cur_tuple, target->arity * sizeof(column_type),
                   cudaMemcpyDeviceToHost);
        for (int j = 0; j < target->arity; j++) {

            std::cout << cur_tuple_host[j] << "     ";
        }
        std::cout << std::endl;
        cudaFreeHost(cur_tuple_host);
    }
    std::cout << "end <<<" << std::endl;

    cudaFreeHost(tuples_host);
}

void print_tuple_raw_data(GHashRelContainer* target, const char *rel_name) {
    column_type* raw_data_host;
    u64 mem_raw = target->data_raw_row_size * target->arity * sizeof(column_type);
    cudaMallocHost((void**) &raw_data_host, mem_raw);
    cudaMemcpy(raw_data_host, target->data_raw, mem_raw, cudaMemcpyDeviceToHost);
    std::cout << "Relation raw tuples >>> " << rel_name << std::endl;
    std::cout << "Total raw tuples counts:  " <<  target->data_raw_row_size << std::endl;
    for (u64 i = 0; i < target->data_raw_row_size; i++) {
        for (int j = 0; j < target->arity; j++) {
            std::cout << raw_data_host[i*target->arity + j] << "    ";
        }
        std::cout << std::endl;
    }
    cudaFreeHost(raw_data_host);
}

//////////////////////////////////////////////////////
// CPU functions

/**
 * @brief load raw data into relation container
 * 
 * @param target hashtable struct in host
 * @param arity 
 * @param data raw data on host
 * @param data_row_size 
 * @param index_columns index columns id in host
 * @param index_column_size 
 * @param index_map_load_factor 
 * @param grid_size 
 * @param block_size 
 */
void load_relation(GHashRelContainer* target, int arity, column_type* data, u64 data_row_size,
                   u64 index_column_size, float index_map_load_factor,
                   int grid_size, int block_size) {
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    // load index selection into gpu
    // u64 index_columns_mem_size = index_column_size * sizeof(u64);
    // checkCuda(cudaMalloc((void**) &(target->index_columns), index_columns_mem_size));
    // cudaMemcpy(target->index_columns, index_columns, index_columns_mem_size, cudaMemcpyHostToDevice);
    if (data_row_size == 0) {
        return;
    }
    // load raw data from host
    u64 relation_mem_size = data_row_size * ((u64)arity) * sizeof(column_type);
    checkCuda(
        cudaMalloc((void **)&(target->data_raw), relation_mem_size)
    );
    cudaMemcpy(target->data_raw, data, relation_mem_size, cudaMemcpyHostToDevice);
    // init tuple to be unsorted raw tuple data address
    checkCuda(cudaMalloc((void**) &target->tuples, data_row_size * sizeof(tuple_type)));
    init_tuples_unsorted<<<grid_size, block_size>>>(target->tuples, target->data_raw, arity, data_row_size);
    // sort raw data
    thrust::sort(thrust::device, target->tuples, target->tuples+data_row_size,
                        tuple_less(index_column_size, arity));
    // print_tuple_rows(target, "after sort");
    
    // deduplication here?
    tuple_type* new_end = thrust::unique(thrust::device, target->tuples, target->tuples+data_row_size,
                                        t_equal(arity));    
    data_row_size = new_end - target->tuples;
    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    // init the index map
    // set the size of index map same as data, (this should give us almost no conflict)
    // however this can be memory inefficient
    target->index_map_size = std::ceil(data_row_size / index_map_load_factor);
    // target->index_map_size = data_row_size;
    u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
    checkCuda(
        cudaMalloc((void**)&(target->index_map), index_map_mem_size)
    );
    
    // load inited data struct into GPU memory
    GHashRelContainer* target_device;
    checkCuda(cudaMalloc((void**) &target_device, sizeof(GHashRelContainer)));
    cudaMemcpy(target_device, target, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);
    init_index_map<<<grid_size, block_size>>>(target_device);
    // std::cout << "finish init index map" << std::endl;
    // print_hashes(target, "after construct index map");
    // calculate hash 
    calculate_index_hash<<<grid_size, block_size>>>(target_device);
    cudaFree(target_device);
}


/**
 * @brief clean all data in a relation container
 * 
 * @param target 
 */
void free_relation(GHashRelContainer* target) {
    cudaFree(target->index_map);
    cudaFree(target->tuples);
    cudaFree(target->data_raw);
}

/**
 * @brief binary join, close to local_join in slog's join RA operator
 * 
 * @param inner 
 * @param outer    
 * @param block_size 
 */
void binary_join(GHashRelContainer* inner, GHashRelContainer* outer,
                 GHashRelContainer* output_newt,
                 int* reorder_array,
                 int reorder_array_size,
                 int grid_size, int block_size) {
    // need copy to device?
    GHashRelContainer* inner_device;
    checkCuda(cudaMalloc((void**) &inner_device, sizeof(GHashRelContainer)));
    cudaMemcpy(inner_device, inner, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);
    GHashRelContainer* outer_device;
    checkCuda(cudaMalloc((void**) &outer_device, sizeof(GHashRelContainer)));
    cudaMemcpy(outer_device, outer, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);

    u64* result_counts_array;
    checkCuda(cudaMalloc((void**) &result_counts_array, outer->tuple_counts * sizeof(u64)));

    // reorder using 2nd column of foo(1),  2nd column of bar(3)
    int* reorder_array_device;
    checkCuda(cudaMalloc((void**) &reorder_array_device, reorder_array_size * sizeof(int)));
    cudaMemcpy(reorder_array_device, reorder_array, reorder_array_size * sizeof(int), cudaMemcpyHostToDevice);
    
    get_join_result_size<<<grid_size, block_size>>>(inner_device, outer_device, 1, result_counts_array);

    // u64* result_counts_array_host;
    // cudaMallocHost((void**) &result_counts_array_host, outer->tuple_counts * sizeof(u64));
    // cudaMemcpy(result_counts_array_host, result_counts_array, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToHost);
    
    u64 total_result_rows = thrust::reduce(thrust::device, result_counts_array, result_counts_array+outer->tuple_counts, 0);
    std::cout << "join result size " << total_result_rows << std::endl;
    u64* result_counts_offset;
    checkCuda(cudaMalloc((void**) &result_counts_offset, outer->tuple_counts * sizeof(u64)));
    cudaMemcpy(result_counts_offset, result_counts_array, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToDevice);
    thrust::exclusive_scan(thrust::device, result_counts_offset, result_counts_offset + outer->tuple_counts, result_counts_offset);

    // u64* result_counts_offset_host;
    // cudaMallocHost((void**) &result_counts_offset_host, outer->tuple_counts * sizeof(u64));
    // cudaMemcpy(result_counts_offset_host, result_counts_offset, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToHost);
    // std::cout << "wwwwwwwwwwwww" <<std::endl;
    // for (u64 i = 0; i < outer->tuple_counts; i++) {
    //     std::cout << result_counts_offset_host[i] << std::endl;
    // }

    column_type* foobar_raw_data;
    checkCuda(cudaMalloc((void**) &foobar_raw_data, total_result_rows * output_newt->arity * sizeof(column_type)));
    get_join_result<<<grid_size, block_size>>>(inner_device, outer_device, 1, reorder_array_device, output_newt->arity,
                                               foobar_raw_data, result_counts_array, result_counts_offset);
    // cudaFree(result_counts_array);

    column_type* foobar_raw_data_host;
    cudaMallocHost((void**) &foobar_raw_data_host, total_result_rows * output_newt->arity * sizeof(column_type));
    cudaMemcpy(foobar_raw_data_host, foobar_raw_data, total_result_rows * output_newt->arity * sizeof(column_type),cudaMemcpyDeviceToHost);
    // std::cout << "wwwwwwwwww" << std::endl;
    // for (u64 i = 0; i < total_result_rows; i++) {
    //     for (int j = 0; j < output_newt->arity; j++) {
    //         std::cout << foobar_raw_data_host[i*output_newt->arity + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // reload newt
    free_relation(output_newt);
    load_relation(output_newt, output_newt->arity, foobar_raw_data, total_result_rows, 1, 0.6, grid_size, block_size);

    cudaFree(inner_device);
    cudaFree(outer_device);
}

//////////////////////////////////////////////////////

// test hash join 
// foobar(a,c) :- foo(a,b), bar(b,c).
// foo(1,2).
// foo(2,3).
// bar(2,4).
// bar(2,4).
// bar(4,4).
// bar(9,3).
// --->
// foobar(3,4).
void foobar_test() {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry" << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    // foo
    column_type foo_data[2*2] = {1,2, 2,3};
    GHashRelContainer* foo_table = new GHashRelContainer();
    int foo_arity = 2;
    load_relation(foo_table, foo_arity, foo_data, 2, 1, 0.6, grid_size, block_size);
    print_hashes(foo_table, "foo");
    print_tuple_rows(foo_table, "foo");

    // bar
    column_type bar_data[4*2] = {2,4, 4,4, 2,4, 9,3};
    GHashRelContainer* bar_table = new GHashRelContainer();
    int bar_arity = 2;
    load_relation(bar_table, bar_arity, bar_data, 4, 1, 0.6, grid_size, block_size);
    print_hashes(bar_table, "bar");
    print_tuple_rows(bar_table, "bar");

    // result relation
    // dummy data
    column_type foobar_dummy_data[2] = {0,0};
    int foobar_arity = 2;
    GHashRelContainer* foobar_table = new GHashRelContainer();
    load_relation(foobar_table, foobar_arity, foobar_dummy_data, 0, 1, 0.6, grid_size, block_size);

    // join
    // use bar as outer relation and foo as inner
    // allocate join result count buffer for each outer tuple
    int reorder_array[2] = {1,3};
    binary_join(foo_table , bar_table, foobar_table, reorder_array, 2, grid_size, block_size);

    print_tuple_rows(foobar_table, "foobar");
    
    // clean and exit
    free_relation(foo_table);
    delete foo_table;
    free_relation(bar_table);
    delete bar_table;
    free_relation(foobar_table);
    delete foobar_table;
}


int main() {
    foobar_test();
    return 0;
}
