#ifndef OMPHELPER
#define OMPHELPER

static inline data_type atomicRead(data_type* const addr)
{
  data_type ret;
  #pragma omp atomic read
  ret = *addr;
  return ret;
}

static inline void atomicWrite(data_type* const addr, const data_type val)
{
  #pragma omp atomic write
  *addr = val;
}

static inline data_type criticalRead(data_type* const addr)
{
  data_type ret;
  #pragma omp critical
  ret = *addr;
  return ret;
}

static inline void criticalWrite(data_type* const addr, const data_type val)
{
  #pragma omp critical
  *addr = val;
}

static inline data_type critical_min(data_type* addr, data_type val)
{
  data_type oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv > val) {
      *addr = val;
    }
  }
  return oldv;
}

static inline data_type critical_max(data_type* addr, data_type val)
{
  data_type oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv < val) {
      *addr = val;
    }
  }
  return oldv;
}

static inline data_type fetch_and_add(data_type* addr)
{
  data_type old;
  #pragma omp atomic capture
  {
    old = *addr;
    (*addr)++;
  }
  return old;
}

static inline data_type criticalAdd(data_type* addr, data_type val)
{
  data_type oldv;
  #pragma omp atomic capture
  {
    oldv = *addr;
    (*addr) += val;
  }
  return oldv;
}
#endif
