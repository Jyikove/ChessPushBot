#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
using namespace std;
struct Point { double x,y; };
double dist(Point a,Point b){return hypot(a.x-b.x,a.y-b.y);}
int main(int argc,char** argv){
    const bool turns=argc>1 && string(argv[1])=="turns";
    int n,target; double A,gx,lo,hi; cout<<setprecision(17);
    while(cin>>n>>target>>A>>gx>>lo>>hi){
        int carried=0; double prefix_before=0,reference=-1;
        if(argc>1) cin>>carried>>prefix_before>>reference;
        double yaw=0,scale=0;if(turns) cin>>yaw>>scale;
        auto bend=[&](double h,double a){return scale*abs(remainder(a-h,2*acos(-1.)));};
        vector<Point> pt(n+1);for(auto &p:pt)cin>>p.x>>p.y;
        vector<double> ends(n+1); vector<vector<double>> legs(n+1,vector<double>(n+1));
        vector<vector<int>> order(n+1); double single=1e99,step=1e99,lastleg=1e99;
        for(int i=0;i<=n;i++){
            ends[i]=dist(pt[i],{gx,clamp(pt[i].y,lo,hi)});
            for(int j=1;j<=n;j++)if(i!=j){legs[i][j]=dist(pt[i],pt[j]);step=min(step,legs[i][j]);order[i].push_back(j);}
            sort(order[i].begin(),order[i].end(),[&](int a,int b){return legs[i][a]<legs[i][b];});
        }
        auto heading=[&](int i,Point p){return atan2(p.y-pt[i].y,p.x-pt[i].x);};
        auto finish=[&](int i,double h){return ends[i]+(turns && ends[i]>1e-10?bend(h,heading(i,{gx,clamp(pt[i].y,lo,hi)})):0);};
        for(int i=1;i<=n;i++){double h=legs[0][i]>1e-10?heading(0,pt[i]):yaw;single=min(single,legs[0][i]+(turns?bend(yaw,h):0)+finish(i,h));lastleg=min(lastleg,ends[i]);}
        if(reference>=0)single=reference;else if(carried)single=finish(0,yaw);
        vector<double> rewards(n+carried+1,0.);
        for(int k=1;k<=n+carried;k++)
            rewards[k]=rewards[k-1]+ldexp(A,-max(0,k-target));
        auto loss=[&](double length,int k){k+=carried;double extra=max(0.,prefix_before+length-single), budget=.24*min(k-1,target-1);return extra+max(0.,extra-budget)-rewards[k];};
        double best=numeric_limits<double>::infinity(),bestlen=0;vector<int> path,answer;
        auto visit=[&](auto&& self,int i,unsigned long long used,double prefix,double h)->void{
            int k=path.size();double length=prefix+finish(i,h);
            if((k || carried) && loss(length,k)<best){best=loss(length,k);bestlen=length;answer=path;}
            bool improve=false;for(int m=k+1;m<=n;m++)if(loss(prefix+max(ends[i],(m-k)*step+lastleg),m)<=best+1e-10){improve=true;break;}
            if(!improve)return;
            for(int j:order[i])if(!(used&(1ULL<<j))){path.push_back(j);double a=turns && legs[i][j]>1e-10?heading(i,pt[j]):h;self(self,j,used|(1ULL<<j),prefix+legs[i][j]+(turns?bend(h,a):0),a);path.pop_back();}
        };visit(visit,0,0,0.,yaw);cout<<best<<' '<<bestlen<<' '<<single;for(int j:answer)cout<<' '<<j;cout<<endl;
    }
}
