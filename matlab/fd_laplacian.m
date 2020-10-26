function [C, M] = fd_laplacian(V, E)
%FD_LAPLACIAN Calculates the discrete laplacian matrix L = M^-1C using finite
%differences
% Inputs
%   V - vertices around closed loop
%   E - Edges of indices onto those vertices
%Outptuts
    [rows, cols] = size(E);
    lengths = edgeLengths(V, E);
    avgL = zeros(rows, 1);
    avgL(E(:, 1)) = lengths(E(:, 1));
    avgL(E(:, 2)) = avgL + lengths(E(:, 2));
    M = diag(avgL)/2;
    D = sparse(E(:, 1), E(:, 2), 1./lengths, rows, rows);
    C = D;
    C = C + C';
    C = C - diag(sum(C, 2))
    
end

function lengths = edgeLengths(V, E)
    [rows, cols] = size(E);
    lengths = zeros(rows, 1);
    displacements = V(E(:, 1), :) - V(E(:, 2), :);
    lengths = transpose(vecnorm(transpose(displacements)));
end
