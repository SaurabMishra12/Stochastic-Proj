
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
static inline uint64_t mix64(uint64_t z) {
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
extern "C" void simulate(int n,int nr,int steps,int delay,int stride,int nsave,
 double dt,double K,double sigma,double rho,double dropout,int threads,
 const double* adjacency,const double* omega,const double* initial,
 const uint64_t* seeds,double* metrics,double* orders,double* samples) {
    int ns=steps/stride+1, histlen=delay+1;
    double sigind=sigma*std::sqrt((1-rho)*dt), sigcom=sigma*std::sqrt(rho*dt);
    int tailstart=(4*steps)/5;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int r=0;r<nr;r++) {
        std::mt19937_64 rng(seeds[r*3+1]); std::normal_distribution<double> normal(0.,1.);
        std::vector<double> x(n),history(histlen*n),drift(n),sins(n),coss(n),z(n),tailx(n),degree(n);
        double global=0.;
        for(int i=0;i<n;i++) global+=initial[r*n+i]/n;
        for(int i=0;i<n;i++) {
            x[i]=initial[r*n+i]-global;
            for(int h=0;h<histlen;h++) history[h*n+i]=x[i];
            degree[i]=0.;for(int j=0;j<n;j++) degree[i]+=adjacency[i*n+j];
        }
        double sumR=0.,sumR2=0.;int ntail=0;
        double finalR=0.;
        for(int s=0;s<=steps;s++) {
            if(s>=tailstart || s%stride==0) {
                double real=0.,imag=0.;
                for(int i=0;i<n;i++){real+=std::cos(x[i]);imag+=std::sin(x[i]);}
                finalR=std::sqrt(real*real+imag*imag)/n;
                if(s>=tailstart){sumR+=finalR;sumR2+=finalR*finalR;ntail++;}
                if(s%stride==0){
                    int k=s/stride; orders[r*ns+k]=finalR;
                    if(r<nsave) for(int i=0;i<n;i++) samples[(r*ns+k)*n+i]=x[i]+global;
                }
            }
            if(s==tailstart) tailx=x;
            if(s==steps)break;
            int oldslot=(s>=delay ? (s-delay)%histlen : 0);
            for(int j=0;j<n;j++){
                sins[j]=std::sin(history[oldslot*n+j]);
                coss[j]=std::cos(history[oldslot*n+j]);
            }
            double driftmean=0.;
            for(int i=0;i<n;i++) {
                double coupling=0.;
                for(int j=0;j<n;j++) if(adjacency[i*n+j]) {
                    bool received=true;
                    if(dropout>0.) {
                        uint64_t key=seeds[r*3+2]^((uint64_t)(s+1)*UINT64_C(0x9e3779b97f4a7c15))
                                     ^((uint64_t)(i*n+j+1)*UINT64_C(0xd1b54a32d192ed03));
                        double u=(mix64(key)>>11)*0x1.0p-53;
                        received=(u>=dropout);
                    }
                    if(received) coupling+=sins[j]*coss[i]-coss[j]*sins[i];
                }
                drift[i]=omega[i]+K*coupling/degree[i]; driftmean+=drift[i]/n;
            }
            double zmean=0.;
            for(int i=0;i<n;i++){z[i]=normal(rng);zmean+=z[i]/n;}
            double shared=normal(rng);
            global+=driftmean*dt+sigind*zmean+sigcom*shared;
            for(int i=0;i<n;i++) x[i]+=(drift[i]-driftmean)*dt+sigind*(z[i]-zmean);
            int slot=(s+1)%histlen;
            for(int i=0;i<n;i++)history[slot*n+i]=x[i];
        }
        double fm=0.,f2=0.,tailduration=(steps-tailstart)*dt;
        for(int i=0;i<n;i++){
            double f=(x[i]-tailx[i])/tailduration;fm+=f/n;f2+=f*f/n;
        }
        double meanR=sumR/ntail,meanR2=sumR2/ntail;
        metrics[r*8]=meanR;
        metrics[r*8+1]=std::max(0.,meanR2-meanR*meanR);
        metrics[r*8+2]=1.-meanR;
        metrics[r*8+3]=(1.-meanR2)*n/(n-1.);
        metrics[r*8+4]=std::sqrt(std::max(0.,f2-fm*fm));
        metrics[r*8+5]=finalR;metrics[r*8+6]=global;metrics[r*8+7]=ntail;
    }
}
