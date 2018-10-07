X = csvread('data_q2.csv');

n=3012;
m=8447;
K=25; % Set the rank to 25
T=100; % run for 100 iterations

% Each value in W and H can be initialized randomly to a positive number, e.g., from a Uniform(1,2) distribution.
w=unifrnd(1,2,3012,K);
h=unifrnd(1,2,K,8447);
para=w*h;
para=max(para,10^-16);
L=zeros(1,T);

for t=1:100
  % update h
  for j = 1:8447
      for k =1:K
          h(k,j)=h(k,j)*(sum(w(:,k).*X(:,j)./para(:,j)))/(max(10^-16,sum(w(:,k))));
      end
  end

    % update w
    para=w*h;
    para=max(para,10^-16);
    for i = 1:3012
        for k = 1:K
            w(i,k)=w(i,k)*(sum(h(k,:).*X(i,:)./para(i,:)))/(max(10^-16,sum(h(k,:))));
        end
    end

    para=w*h;
    para=max(para,10^-16);
    % objective func
    % [row, col] = find(isnan(ans));
    L(t)= sum(nansum(para-(X.*log(para))));
end



% normalize the columns of W
for i=1:K
  w[i]=w[i]/sum(w[i],1);
end
