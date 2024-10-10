function  labels = ETEAL(X,gt,numanchor,param,dim_H)
%%
nv=length(X); % view num
num=length(gt); % sample num
ns=length(unique(gt));
Sbar = [];
alpha = param.alpha;
lambda = param.lambda;
sX = [numanchor,num, nv];

%% Initialization
 for iv=1:nv
     H{iv} = zeros(dim_H,num);
     B{iv} = zeros(size(X{iv},1),dim_H);
     A{iv} = zeros(dim_H,numanchor);
     Zi{iv} = zeros(numanchor, num);
     Y{iv} = zeros(numanchor, num);
     L{iv} = zeros(numanchor, num);
 end

%%
Isconverg = 0;
iter = 0;
mu = 1e-3;
dt = 1.3;

%% 
 while(Isconverg==0)
%% -------------------update H--------------------------
   for iv=1:nv
        H_a =lambda* eye(dim_H) + B{iv}'*B{iv};
        H_b = B{iv}'*X{iv} + lambda*A{iv}*Zi{iv};
        H{iv} = H_a\H_b;  
   end
%% -------------------update B--------------------------
   for iv=1:nv
        W= X{iv}*H{iv}';      
        [M,~,P] = svd(W,'econ');
        B{iv} = M*P';
   end
%% -------------------update Z--------------------------
    for iv = 1:nv
            Z = zeros(numanchor, num);
            Q = ( 2*lambda*A{iv}'*H{iv}-Y{iv}+mu*L{iv} )/(2*lambda+ mu);
            for ji=1:num
                idx = 1:ns;
                Z(idx, ji) = EProjSimplex_new(Q(idx,ji));
            end
            Zi{iv}=Z;
    end
%% ------------------update A---------------------------
    for iv = 1:nv
            HZ = lambda*H{iv} * Zi{iv}';
            [AU,~,AV] = svd(HZ, 'econ');
            A{iv} = AU * AV';
    end
%% ------------------update L---------------------------
    Zi_tensor = cat(3, Zi{:,:});
    Y_tensor = cat(3, Y{:,:});
    Ziv = Zi_tensor(:);
    Yv = Y_tensor(:);
    [Lv, ~] = wshrinkObj_nc(param.fun, param.gamma,Ziv + 1/mu*Yv, alpha/mu, [numanchor, num, nv], 0, 3);
    L_tensor = reshape(Lv, [numanchor, num, nv]); 
    for iv=1:nv
        L{iv} = L_tensor(:,:,iv);
    end     
%% ------------------update Y---------------------------
    for iv=1:nv
        Y{iv} = Y{iv} + mu*(Zi{iv} - L{iv});
    end
     mu = min(1e6, dt*mu);

%% 
   
  if (iter>50) 
         Isconverg = 1;
  end
    iter = iter + 1; 

 Sbar= [];
for iv=1:nv
        rowsum = sum(L{iv},2);
        rowsum_sq = rowsum.^(-1/2);
        rowsum_sq(rowsum_sq==inf)=0;
        sigma_sq = diag(rowsum_sq);
        Lbar = sigma_sq * L{iv}; 
        Sbar = cat(1, Sbar, Lbar); 
end
 end
[U,~,~] = mySVD(Sbar',ns); 
rand('twister',5489)
labels=litekmeans(U, ns, 'MaxIter', 50,'Replicates',10);
