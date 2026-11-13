#include <cmath>
#include <cstdint>
extern "C" void pll_path(const double* dw,int n,double h,double a,double b,double delta,double alpha,double phi,double v,int stride,double* path,double* summary,double* event_times,int* event_dirs,int* nevents){
 const double pi=std::acos(-1.0), twopi=2*pi;
 bool wells=std::abs(delta/(a*b))<1.;
 double stable=wells?std::asin(delta/(a*b)):0.;
 int occupied[3],candidate[3]; double age[3]={0,0,0}; int counts[3]={0,0,0};
 double radius[3]={pi/4,pi/2,pi/6},dwell[3]={.5,.25,1.};
 for(int r=0;r<3;r++){occupied[r]=(int)std::floor((phi-stable)/twopi+.5);candidate[r]=occupied[r];}
 int forward=0,backward=0,ev=0,tailn=0;double tailv=0,tailmean=0,tailm2=0,tailerr=0,first=-1;
 path[0]=phi;path[1]=v;int sample=1; double tailstartphi=phi;
 for(int i=0;i<n;i++){
   double oldphi=phi;
   phi+=v*h;
   v+=(delta-v-a*b*std::sin(oldphi))/a*h-alpha/a*dw[i];
   double time=(i+1)*h;
   if((i+1)%stride==0){path[2*sample]=phi;path[2*sample+1]=v;sample++;}
   if(i==n*4/5-1)tailstartphi=phi;
   if(i>=n*4/5){tailn++;tailv+=v;double d=phi-tailmean;tailmean+=d/tailn;tailm2+=d*(phi-tailmean);tailerr+=std::abs(std::remainder(phi-stable,twopi));}
   if(wells){
    int k=(int)std::floor((phi-stable)/twopi+.5);
    for(int r=0;r<3;r++){
     if(k!=occupied[r] && std::abs(phi-stable-twopi*k)<=radius[r]){
      int target=occupied[r]+(k>occupied[r]?1:-1);
      if(target!=k){age[r]=0;continue;}
      if(candidate[r]!=target){candidate[r]=target;age[r]=0;}
      age[r]+=h;
      if(age[r]+1e-12>=dwell[r]){
       int direction=target-occupied[r];counts[r]++;occupied[r]=target;candidate[r]=target;age[r]=0;
       if(r==0){event_times[ev]=time;event_dirs[ev++]=direction;if(first<0)first=time;if(direction>0)forward++;else backward++;}
      }
     }else{candidate[r]=occupied[r];age[r]=0;}
    }
   }
 }
 summary[0]=phi;summary[1]=v;summary[2]=tailv/tailn;summary[3]=tailm2/(tailn>1?tailn-1:1);summary[4]=tailerr/tailn;
 summary[5]=wells?forward:NAN;summary[6]=wells?backward:NAN;summary[7]=first;
 summary[8]=wells?counts[1]:NAN;summary[9]=wells?counts[2]:NAN;
 bool captured=wells && std::abs(std::remainder(phi-stable,twopi))<pi/4 && std::abs(summary[2])<.1;
 summary[10]=wells?(captured?1:0):NAN;summary[11]=wells?((captured && counts[0]==0)?1:0):NAN;
 summary[12]=(phi-tailstartphi)/(.2*n*h)/twopi;
 *nevents=ev;
}
