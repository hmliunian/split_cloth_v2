#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
// Full endpoint-to-endpoint DTW. At most three consecutive horizontal/vertical
// moves; progress band +-25%. Sum-minimizing path, normalized by its length.
extern "C" double dtw32(const double* x, const double* y, int dims) {
  constexpr int N=32, S=7;
  double cost[N*N*S]; int count[N*N*S];
  std::fill(cost,cost+N*N*S,std::numeric_limits<double>::infinity());
  std::fill(count,count+N*N*S,0);
  auto idx=[](int i,int j,int s){return (i*N+j)*S+s;};
  for(int i=0;i<N;i++) for(int j=0;j<N;j++) {
    if(std::abs(i-j)>8) continue;
    double d=0;for(int k=0;k<dims;k++){double z=x[i*dims+k]-y[j*dims+k];d+=z*z;}
    if(i==0&&j==0){cost[0]=d;count[0]=1;continue;}
    auto put=[&](int target,int prior){
      if(cost[prior]+d<cost[target]){cost[target]=cost[prior]+d;count[target]=count[prior]+1;}
    };
    if(i>0&&j>0)for(int s=0;s<S;s++)put(idx(i,j,0),idx(i-1,j-1,s));
    if(i>0){
      for(int s: {0,4,5,6})put(idx(i,j,1),idx(i-1,j,s));
      for(int s=2;s<=3;s++)put(idx(i,j,s),idx(i-1,j,s-1));
    }
    if(j>0){
      for(int s: {0,1,2,3})put(idx(i,j,4),idx(i,j-1,s));
      for(int s=5;s<=6;s++)put(idx(i,j,s),idx(i,j-1,s-1));
    }
  }
  int best=idx(N-1,N-1,0);
  for(int s=1;s<S;s++)if(cost[idx(N-1,N-1,s)]<cost[best])best=idx(N-1,N-1,s);
  return cost[best]/count[best];
}
