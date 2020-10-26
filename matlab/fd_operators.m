function [D,M, N] = fd_operators(V)
% Builds 'D' and 'M' (lumped) as in finite differences for a closed 
% curve parametrization
%
%   Input:
%           - V #V by dim vertex list (ordered).
%   Output:
%           - D #V by #V matrix of un-integrated Laplacian
%           - M #V by #V matrix s.t. L = inv(M)*D
%
% Mostly used in curvature_flow.m
n = size(V,1);
                edges = [V(2:end,:);V(1,:)]-V;
                h = transpose(vecnorm(transpose(edges)));
                vedges = ( h+[h(n);h(1:(n-1))] )./2;
                E = 1./h;
                M = diag(vedges);
                N = diag(h);
                D = sparse(1:n,[2:n,1],E,n,n);
%                 G = D-D';
                G = D-diag(sum(D,2));
                D = D+D';
                D = D-diag(sum(D,2));
end

