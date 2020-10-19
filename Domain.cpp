#include "Domain.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/boundary_loop.h>
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <igl/unique.h>
#include <igl/marching_tets.h>
#include <igl/isolines.h>
#include <igl/remesh_along_isoline.h>
#include <igl/colon.h>
#include <igl/setdiff.h>
#include <igl/point_simplex_squared_distance.h>

Eigen::MatrixXd make2D(Eigen::MatrixXd mat)
{
	Eigen::MatrixXd mat2D(mat.rows(), 2);
	mat2D.col(0) = mat.col(0);
	mat2D.col(1) = mat.col(1);
	return mat2D;
}

Eigen::MatrixXd make3D(Eigen::MatrixXd mat)
{
	Eigen::MatrixXd mat3D(mat.rows(), 3);
	mat3D.setZero();
	mat3D.col(0) = mat.col(0);
	mat3D.col(1) = mat.col(1);
	return mat3D;

}
void Domain::diffuseHeat(float dt)
{
	Eigen::VectorXi boundaryIndices;
	Eigen::VectorXd  boundaryTemperatures;

	boundaryIndices.resize(Boundary.Vi.rows() + I.Vi.rows(), 1);
	boundaryIndices << Boundary.Vi, I.Vi;
	boundaryTemperatures.resize(boundaryIndices.rows(), 1);
	boundaryTemperatures << Boundary.T, I.T;

	Eigen::SparseMatrix<double> C, M, Q; //cotan matrix, identity matrix, quadratic coefficient term
	Eigen::VectorXd B = Eigen::VectorXd::Zero(V.rows(), 1); // linear coefficients



	igl::cotmatrix(V, F, C);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	B = -M * T; //Linear coefficients are equal to last timesteps solved Temperature field
	B.setZero(); // set B to zero if using steady state approx
	//Q = M - dt * C;
	Q = C; //set Q to simply C if using steady state approx
	igl::min_quad_with_fixed_data<double> mqwf;
	Eigen::VectorXd Beq;
	Eigen::SparseMatrix<double> Aeq;
	igl::min_quad_with_fixed_precompute(Q, boundaryIndices, Aeq, false, mqwf);
	igl::min_quad_with_fixed_solve(mqwf, B, boundaryTemperatures, Beq, T);

};


void Domain::melt(float dt)
{

	V += VertexVel *  dt;


}


void Domain::solveForVertexVelSmart()
{
//	fillL();
	fillM();
	fillA();

	Eigen::VectorXd vp = flattenMat2Vec(Vp);
	//Eigen::MatrixXd Vp2 = expandVec2Mat(vp, 2);
	Eigen::VectorXd vpn = projectVecs2Normals(Vp, NormalsE );
	Eigen::VectorXd b = I.A*vpn;
	Eigen::VectorXd v;
	/*
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
	solver.compute(I.M);
	v = solver.solve(b);
	*/
	Eigen::SparseMatrix<double> eye;
	eye.resize(I.M.rows(), I.M.cols());
	eye.setIdentity();
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
	solver.compute(I.M + lambda * eye);
	v = solver.solve(b);




	VertexVel = expandVec2Mat(v, 3);
	//VertexVel = make3D(VertexVel);



}

void Domain::fillM()
{
	typedef Eigen::Triplet<double> T;
	std::vector<T> triplets;
	I.M.resize(3 * V.rows(), 3 * V.rows());//(3*I.Vi.rows(), 3*I.Vi.rows()); // M is techincally |Vb| by |Vb| where Vb is the number of vertices on boundary
	I.M.setZero();
	double length;
	int v1i, v2i;
	double sixth = 1.0 / 6.0;
	Eigen::Vector3d normal;
	double nx, ny, nz, nx2, nz2, ny2, nxny, nxnz, nynz;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = (I.E(i, 0)); //global
		v2i = (I.E(i, 1));
		normal = I.NormalsE.row(i);
		nx = normal(0); ny = normal(1); nz = normal(2);
		nx2 = nx * nx; ny2 = ny * ny; nz2 = nz * nz; nxny = nx * ny; nxnz = nx * nz; nynz = ny * nz;
		length = edgeLength(v1i, v2i);

		//Top Left Corner
		triplets.push_back(T(3 * v1i + 0, 3 * v1i + 0, 2.0 * nx2 * length * sixth));
		triplets.push_back(T(3 * v1i + 0, 3 * v1i + 1, 2.0 * nxny * length * sixth));
		triplets.push_back(T(3 * v1i + 0, 3 * v1i + 2, 2.0 * nxnz * length * sixth));

		triplets.push_back(T(3 * v1i + 1, 3 * v1i + 0, 2.0 * nxny * length * sixth));
		triplets.push_back(T(3 * v1i + 1, 3 * v1i + 1, 2.0 * ny2 * length * sixth));
		triplets.push_back(T(3 * v1i + 1, 3 * v1i + 2, 2.0 * nynz * length * sixth));

		triplets.push_back(T(3 * v1i + 2, 3 * v1i + 0, 2.0 * nxnz * length * sixth));
		triplets.push_back(T(3 * v1i + 2, 3 * v1i + 1, 2.0 * nynz * length * sixth));
		triplets.push_back(T(3 * v1i + 2, 3 * v1i + 2, 2.0 * nz2 * length * sixth));


		//Top Right Corner
		triplets.push_back(T(3 * v1i + 0, 3 * v2i + 0, nx2 * length * sixth ));
		triplets.push_back(T(3 * v1i + 0, 3 * v2i + 1, nxny * length * sixth));
		triplets.push_back(T(3 * v1i + 0, 3 * v2i + 2, nxnz * length * sixth));

		triplets.push_back(T(3 * v1i + 1, 3 * v2i + 0, nxny * length * sixth));
		triplets.push_back(T(3 * v1i + 1, 3 * v2i + 1, ny2 * length * sixth ));
		triplets.push_back(T(3 * v1i + 1, 3 * v2i + 2, nynz * length * sixth));

		triplets.push_back(T(3 * v1i + 2, 3 * v2i + 0, nxnz * length * sixth));
		triplets.push_back(T(3 * v1i + 2, 3 * v2i + 1, nynz * length * sixth));
		triplets.push_back(T(3 * v1i + 2, 3 * v2i + 2, nz2 * length * sixth ));

		//Bottom Left Corner
		triplets.push_back(T(3 * v2i + 0, 3 * v1i + 0, nx2 * length * sixth));
		triplets.push_back(T(3 * v2i + 0, 3 * v1i + 1, nxny * length * sixth));
		triplets.push_back(T(3 * v2i + 0, 3 * v1i + 2, nxnz * length * sixth));

		triplets.push_back(T(3 * v2i + 1, 3 * v1i + 0, nxny * length * sixth));
		triplets.push_back(T(3 * v2i + 1, 3 * v1i + 1, ny2 * length * sixth));
		triplets.push_back(T(3 * v2i + 1, 3 * v1i + 2, nynz * length * sixth));

		triplets.push_back(T(3 * v2i + 2, 3 * v1i + 0, nxnz * length * sixth));
		triplets.push_back(T(3 * v2i + 2, 3 * v1i + 1, nynz * length * sixth));
		triplets.push_back(T(3 * v2i + 2, 3 * v1i + 2, nz2 * length * sixth));


		//Bottom Right Corner
		triplets.push_back(T(3 * v2i + 0, 3 * v2i + 0, 2.0 * nx2 * length * sixth));
		triplets.push_back(T(3 * v2i + 0, 3 * v2i + 1, 2.0 * nxny * length * sixth));
		triplets.push_back(T(3 * v2i + 0, 3 * v2i + 2, 2.0 * nxnz * length * sixth));
							
		triplets.push_back(T(3 * v2i + 1, 3 * v2i + 0, 2.0 * nxny * length * sixth));
		triplets.push_back(T(3 * v2i + 1, 3 * v2i + 1, 2.0 * ny2 * length * sixth));
		triplets.push_back(T(3 * v2i + 1, 3 * v2i + 2, 2.0 * nynz * length * sixth));
							
		triplets.push_back(T(3 * v2i + 2, 3 * v2i + 0, 2.0 * nxnz * length * sixth));
		triplets.push_back(T(3 * v2i + 2, 3 * v2i + 1, 2.0 * nynz * length * sixth));
		triplets.push_back(T(3 * v2i + 2, 3 * v2i + 2, 2.0 * nz2 * length * sixth));

	}
	I.M.setFromTriplets(triplets.begin(), triplets.end());


}

void Domain::fillA()
{
	typedef Eigen::Triplet<double> T;
	std::vector<T> triplets;
	I.A.resize(3 * V.rows(),  EV.rows()); //(3 * I.Vi.rows(), 3 * I.Ei.rows());
	I.A.setZero();
	double length;
	int v1i, v2i;
	Eigen::Vector3d normal;
	double nx, ny, nz;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = I.E(i, 0);
		v2i = I.E(i, 1);
		length = edgeLength(v1i, v2i);

		normal = I.NormalsE.row(i);
		nx = normal(0); ny = normal(1); nz = normal(2);

		triplets.push_back(T(3 * v1i + 0, I.Ei(i), length * nx * 0.5));
		triplets.push_back(T(3 * v1i + 1, I.Ei(i), length * ny * 0.5));
		triplets.push_back(T(3 * v1i + 2, I.Ei(i), length * nz * 0.5));

		triplets.push_back(T(3 * v2i + 0, I.Ei(i), length * nx * 0.5));
		triplets.push_back(T(3 * v2i + 1, I.Ei(i), length * ny * 0.5));
		triplets.push_back(T(3 * v2i + 2, I.Ei(i), length * nz * 0.5));
	}
	I.A.setFromTriplets(triplets.begin(), triplets.end());
	
}

void Domain::fillMVLSE()
{

	I.M.resize(3*V.rows(), 3*V.rows());//(3*I.Vi.rows(), 3*I.Vi.rows()); // M is techincally |Vb| by |Vb| where Vb is the number of vertices on boundary
	I.M.setZero();
	float length;
	int v1i, v2i;
	float sixth = 1.0f / 6.0f;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = (I.E(i, 0)); //global
		v2i = (I.E(i, 1));
		length = edgeLength(v1i, v2i);
		//v1i = I.globalToLocalV(v1i); //local
		//v2i = I.globalToLocalV(v2i);

		I.M.coeffRef(3 * v1i + 0, 3 * v2i + 0) += length* sixth;
		I.M.coeffRef(3 * v1i + 1, 3 * v2i + 1) += length* sixth;
		I.M.coeffRef(3 * v1i + 2, 3 * v2i + 2) += length* sixth;
														
		I.M.coeffRef(3 * v2i + 0, 3 * v1i + 0) += length* sixth;
		I.M.coeffRef(3 * v2i + 1, 3 * v1i + 1) += length* sixth;
		I.M.coeffRef(3 * v2i + 2, 3 * v1i + 2) += length* sixth;
														
		I.M.coeffRef(3 * v1i + 0, 3 * v1i + 0) += length* sixth*2.0;
		I.M.coeffRef(3 * v1i + 1, 3 * v1i + 1) += length* sixth*2.0;
		I.M.coeffRef(3 * v1i + 2, 3 * v1i + 2) += length* sixth*2.0;
															 
		I.M.coeffRef(3 * v2i + 0, 3 * v2i + 0) += length* sixth*2.0;
		I.M.coeffRef(3 * v2i + 1, 3 * v2i + 1) += length* sixth*2.0;
		I.M.coeffRef(3 * v2i + 2, 3 * v2i + 2) += length* sixth*2.0;
	}
	Eigen::MatrixXd MD(I.M);

}

void Domain::fillAVLSE()
{

	I.A.resize(3 * V.rows(), 3 * EV.rows()); //(3 * I.Vi.rows(), 3 * I.Ei.rows());
	I.A.setZero();
	float length;
	int v1i, v2i;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = I.E(i, 0);
		v2i = I.E(i, 1);
		length = edgeLength(v1i, v2i);
	//	v1i = I.globalToLocalV(v1i); //local
	//	v2i = I.globalToLocalV(v2i);
		I.A.coeffRef(3 * v1i + 0, 3 * I.Ei(i) + 0) += length * 0.5;
		I.A.coeffRef(3 * v1i + 1, 3 * I.Ei(i) + 1) += length * 0.5;
		I.A.coeffRef(3 * v1i + 1, 3 * I.Ei(i) + 2) += length * 0.5;
					 				   		
		I.A.coeffRef(3 * v2i + 0, 3 * I.Ei(i) + 0) += length * 0.5;
		I.A.coeffRef(3 * v2i + 1, 3 * I.Ei(i) + 1) += length * 0.5;
		I.A.coeffRef(3 * v1i + 2, 3 * I.Ei(i) + 2) += length * 0.5;
	}

	Eigen::MatrixXd AD(I.A);
}


void Domain::interface2GlobalValues()
{
	MidPE.resize(EV.rows(), 3);
	NormalsE.resize(EV.rows(), 3);
	Vp.resize(EV.rows(), 3);
	MidPE.setZero();
	NormalsE.setZero();
	Vp.setZero();
	for (int i = 0; i < I.E.rows(); i++)
	{
		MidPE.row(I.Ei(i)) = I.MidPE.row(i);
		NormalsE.row(I.Ei(i)) = I.NormalsE.row(i);
		Vp.row(I.Ei(i)) = I.Vp.row(i);
	}
}

void Domain::distributeEdgeVpToVertices()
{
	VertexVel.resize(V.rows(), 3);
	VertexVel.setZero();
	int v1i, v2i;
	Eigen::Vector3d vp;		//Incident edge's' velocity
	float length = 0;
	float dot1, dot2;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = I.E(i, 0);
		v2i = I.E(i, 1);

		vp = I.Vp.row(i);
		length = edgeLength(v1i, v2i);

		dot1 = 0.0;// vp.normalized().dot(VertexVel.row(v1i).normalized());
		dot2 = 0.0;// vp.normalized().dot(VertexVel.row(v2i).normalized());
		VertexVel.row(v1i) += length * vp * (1.0 - dot1);
		VertexVel.row(v2i) += length * vp * (1.0 - dot2);
	}
}

float Domain::edgeLength(int v1i, int v2i)
{
	return (V.row(v1i) - V.row(v2i)).norm();
}


void Domain::calculateQuantities(float latentHeat)
{
	calculateGradient(); //calculate and fill FGRAD
	calculateInterfaceMidpoints();
	calculateInterfaceNormals();
	calculateInterfaceVp(latentHeat);

	interface2GlobalValues(); //move everything to the global domain

	solveForVertexVelSmart();
}


void Domain::calculateInterfaceNormals()
{
	I.NormalsE.resize(I.E.rows(), 3);
	int v1i, v2i, fsi, fli;
	Eigen::Vector3d d, n, screenN;
	screenN << 0.0, 0.0, 1.0;
	Eigen::Vector3d bcS, bcL;
	Eigen::Vector3d correctDir;
	for (int i = 0; i < I.E.rows(); i++)
	{
		fsi = I.FiS(i); //boudnary edge adjacent faces
		fli = I.FiL(i);
		v1i = I.E(i, 0); //indices of vertices
		v2i = I.E(i, 1);
		
		bcS = BC.row(fsi);
		bcL = BC.row(fli);
		correctDir = (bcL - bcS).normalized();//direction used to ensure well directed normal

		//get normal of boundary edge (poitns from solid to liquid)
		d = (V.row(v1i) - V.row(v2i)).normalized();
		n = d.cross(screenN).normalized();
		n = (n.dot(correctDir) > 0.0f) ? n : -1.0 *n;//make sure n is pointing in correct direction(From S to L)
		I.NormalsE.row(i) = n;

	}
}

void Domain::calculateInterfaceMidpoints()
{
	I.MidPE.resize(I.E.rows(), 3);
	int v1i, v2i;
	Eigen::Vector3d d;
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = I.E(i, 0); //indices of vertices
		v2i = I.E(i, 1);
		
		d = V.row(v1i) + V.row(v2i);
		d /= 2.0f;
		I.MidPE.row(i) = d;
	}
}

void Domain::calculateInterfaceVp(float latentHeat)
{
	I.Vp.resize(I.E.rows(), 3);
	
	int fsi, fli;
	Eigen::Vector3d  n;
	Eigen::Vector3d sGrad, lGrad, vp;
	for (int i = 0; i < I.E.rows(); i++)
	{
		fsi = I.FiS(i); //boudnary edge adjacent faces
		fli = I.FiL(i);

		n = I.NormalsE.row(i);
		
		//get gradient at both faces
		sGrad = FGrad.row(fsi);
		lGrad = FGrad.row(fli);

		//apply stefan condition
		vp = ((sGrad.dot(n) - lGrad.dot(n)) / latentHeat)*n;
		I.Vp.row(i) = vp;

	}
}



void Domain::calculateGradient()
{
	Eigen::SparseMatrix<double> G;  //Gradient operator. Takes scalar (T) stored at each vertex, returns gradient of T at each face.
	igl::grad(V, F, G);	//Get gradient operator

	FGrad = Eigen::Map<const Eigen::MatrixXd>((G*T).eval().data(), F.rows(), 3); //Gradient of T on each face.

}

bool Domain::isBoundaryVertex(Eigen::Vector3d v)
{
	return (v(0) == maxX || v(0) == minX || v(1) == maxY || v(1) == minY);

}

Eigen::VectorXd flattenMat2Vec(Eigen::MatrixXd mat)
{
	Eigen::VectorXd vec(mat.rows()*mat.cols());
	vec.setZero();
	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
		{
			vec(mat.cols()*i + j) = mat(i, j);
		}
	}

	return vec;
}



Eigen::MatrixXd expandVec2Mat(Eigen::VectorXd vec, int cols)
{
	assert(vec.rows() % cols == 0);
	Eigen::MatrixXd mat(vec.rows() / cols, cols);
	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mat(i, j) = vec(cols*i + j);
		}
	}

	return mat;
}

void Domain::calculateInterpolationAlongEdges()
{
	int numSamples = 15; //number of poitns to sample throughout edge (not including endpoints)
	int v1i, v2i;
	Eigen::Vector3d v1, v2; //Two vertices with which we will interpolate
	Eigen::Vector3d vel1, vel2, velInt, xInt, edge;
	double length, sampleSpacing, s;

	I.intV.resize((numSamples + 2) * I.E.rows(), 3);
	I.intX.resize((numSamples + 2) * I.E.rows(), 3);

	int counter = 0;
	//Loop through edges
	for (int i = 0; i < I.E.rows(); i++)
	{
		v1i = I.E(i, 0);
		v2i = I.E(i, 1);
		v1 = V.row(v1i);
		v2 = V.row(v2i);
		vel1 = VertexVel.row(v1i);
		vel2 = VertexVel.row(v2i);
		edge = v2 - v1; 
		length = edge.norm();
		
		s = 0.0;
		for (int j = 0; j < numSamples + 2; j++)
		{
			xInt = (1.0 - s)*v1 + s * v2;
			velInt = (1.0 - s)*vel1 + s * vel2;
			I.intV.row(counter) = velInt;
			I.intX.row(counter) = xInt;
			counter++;
			s += 1.0 / (numSamples + 2.0);
		}

	}

}

Eigen::VectorXd Domain::projectVecs2Normals(Eigen::MatrixXd& vecs, Eigen::MatrixXd& normals)
{
	Eigen::Vector3d vec;
	Eigen::Vector3d normal;
	Eigen::VectorXd projections(vecs.rows());
	for (int i = 0; i < vecs.rows(); i++)
	{
		vec = vecs.row(i);
		normal = normals.row(i);
		projections(i) = normal.transpose()*vec;
	}

	return projections;
}

