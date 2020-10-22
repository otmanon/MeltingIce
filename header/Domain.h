#pragma once
#include "eigen/Dense"
#include "eigen/Sparse"
/*
Helper function that flattens an |N|x2 matrix into an |2N|x1 vector
*/
Eigen::VectorXd flattenMat2Vec(Eigen::MatrixXd mat);

/*
Helper function to take an |2N|x1 vector and expand into a |N|x2 matrix
*/
Eigen::MatrixXd expandVec2Mat(Eigen::VectorXd vec, int cols);

struct SubDomain
{
	//Domain* parentDomain;//poitner to parent domain
	Eigen::MatrixXi F; // Faces belonging to that domain . The indices in the faces refer to vertices in the parent Domains V matrix( including interface inclusive)
	Eigen::VectorXd T; // Temperatures associated to the vertices of this domain
	Eigen::VectorXi Vi;// Indices from parentDomain's Vertices list that belongs to this domain. (excluding interface indices)

};

struct Interface
{
	//	Domain* parentDomain; //poitner to parent domain
	Eigen::MatrixXi E; // Edges belonging to the interface. The indices in the edges refer to vertices in the parent Domains V matrix. This is ordered
	Eigen::VectorXi Ei; // indices mapping boundary edges to edges in global domain. This is ordered
	Eigen::VectorXd T; // Temperatures associated to the vertices in this interface. 
	Eigen::VectorXi Vi;// Indices from parentDomain's Vertices list that belongs to this domain

	Eigen::VectorXi FiS; //Vector of indices into solid face list to which this edge belongs to. Follows order of E, Ei, 
	Eigen::VectorXi FiL; //Vector of indices into liquid face list to which this edge belongs to Follows order of E, Ei


	Eigen::MatrixXd NormalsE;//Normals for each edge, pointing from solid to liquid, follows order of E, Ei.
	Eigen::MatrixXd MidPE;	//MidPoints for each edge, primarily for visualization. Follows order of E, Ei
	Eigen::MatrixXd Vp;		//desired Velocity for each edge as calculated by stefan position

	//This LoopE is only here because for some reason igl::winding_number needs the edges in their right "loop" order
	Eigen::MatrixXi LoopE; //Edges belonging to interface, but organized in loop order, ie edge i precedes edge i+1 etc. Different order from E, Ei
	Eigen::VectorXi LoopVIndices;	//entries i will hold i'th vertex around loop. 
	Eigen::MatrixXi LoopVPositions; //will hold positions for each vertex i. ordered with loopVIndi

	Eigen::VectorXi globalToLocalV;
	Eigen::VectorXi localToGlobalV;

	Eigen::VectorXi globalToLocalE;
	Eigen::VectorXi localToGlobalE;

	Eigen::MatrixXd MidV;//contains midpoitns of all boundary edges

	Eigen::MatrixXd intX;	//interpolated positions
	Eigen::MatrixXd intV;	//interpolated velocity

	Eigen::SparseMatrix<double> M; //mass matrix 
	Eigen::SparseMatrix<double> A; //A matrix
	Eigen::SparseMatrix<double> L; //L matrix
private:

};

struct Domain
{
	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd V;	//mesh geometry for solid
	Eigen::MatrixXi F;
	Eigen::MatrixXd BC;

	Eigen::MatrixXi EV;  //Matrix of edges (Ne, 2), where each row is an edge, whose entries are both vertex indices
	Eigen::MatrixXi FE;  //Matrix of face-edge relation (Nf, 3) each row is a face and each column are the edge indices that belong to that face
	Eigen::MatrixXi EF;  //Matrix of edge-face relation (Ne, 2) each row is an edge and each column are the face indices that belong to that edge

	Eigen::MatrixXd FGrad;	//Gradient on faces 
	Eigen::MatrixXd VertexVel; //Velocity of vertex point as determined according to desired velocity at incidient edges

	Eigen::VectorXd T; // Temperature at each vertex
	Eigen::VectorXd Tb;	// temperature at each boundary vertex


	
	Eigen::VectorXd W; //Winding number...

	Eigen::MatrixXd NormalsE;	//Normals for each edge, pointing from solid to liquid,
	Eigen::MatrixXd MidPE;		//MidPoints for each edge, primarily for visualization.
	Eigen::MatrixXd Vp;			//desired Velocity for each edge as calculated by stefan position

	Interface I;
	Interface Boundary;
	SubDomain S;
	SubDomain L;


	float lambda = 0.01;

	float maxX, maxY, minX, minY;

	/*
	Diffuses heat by solving the heat equation by minmizing a semi-implicit time update of the Energy Functional :
	(u_n+1)^T (D.I - dt * C)u_n+1 -  (u_n+1)^T(u_n)

	Inputs:
	V: Nx3 Matrix of vertices
	F: Mx3 Matrix of faces
	bIndices : Vector containing indices of boundary
	TBoundary: Vector containing boundary conditions at each boundary vertex

	Output:
	T : Temperature at the current timestep, which will be overwritten as the quadratic energy min. takes effect.
	*/
	void diffuseHeat(float dt);

	/*
	Calculates the gradient of the temperature field near the surface.
	Then, moves the surface inwards based on the stefan condition:
	Lv = ks dTs/dn - kl dTl/dn
	*/
	void melt(float dt);

	/*
	Given gradient of scalar field accross each vertex, calculates the gradient at each vertex (average from neighborin gfaces)
	*/
	void calculateGradient();


	/*
	Calculates and fills NormalsE, with the normal at the boudnary edge with that corresponding vertex
	*/
	void calculateInterfaceNormals();

	/*
	Calculats and fills MidPE, the midpoint at each edge
	*/
	void calculateInterfaceMidpoints();

	/*
	Calculates Vp at each edge, set according to the Stefan Condition.
	*/
	void calculateInterfaceVp(float latentHeat);

	/*
	Helpfer function that calls all other calculation functions required prior to melting
	*/
	void calculateQuantities(float latentHeat);

	/*
	Distributes each edge's VP to the vertices by weighting them with their area. Will try to improve upon this
	*/
	void distributeEdgeVpToVertices();

	/*
	Helper function, returns length of an edge, given two vertex indices in V list.
	*/
	float edgeLength(int v1i,int v2i);

	/*
	Solves for vertex vel by using minimization of error accross boudnary
	*/
	void solveForVertexVelSmart();

	/*
	Solves for vertex vel by using minimization of error accross boudnary, but with simplified way
	*/
	void solveForVertexVelOkay();

	/*
	Fills M matrix using the better normal projection LSE
	*/
	void fillM();

	/*
	Fills A matrix using the better normal projection LSE
	*/
	void fillA();

	/*
	Fills mass matrix using the "wrong" vector LSE
	*/
	void fillMVLSE();

	/*
	Fills L matrix
	*/
	void fillL();

	/*
	fills A Matrix using the "wrong" vector LSE
	*/
	void fillAVLSE();

	/*
	Takes values calculated on interface such as Vp, normals and midpoitns, and puts them in the global array
	*/
	void interface2GlobalValues();

	bool isBoundaryVertex(Eigen::Vector3d v);

	/*
	For each edge, sample 10 points along the edge and interpolate the velocity at that point so that 
	this could be displayed for debugging purposes.
	*/
	void calculateInterpolationAlongEdges();

	/*
	Projects list of vectors to a corresponding list of normals
	*/
	Eigen::VectorXd projectVecs2Normals(Eigen::MatrixXd& vectors, Eigen::MatrixXd& normals);

	/*
	Goes through interface vertices and if they are too close, make them one
	*/
	void mergeInterfaceVertices(Eigen::MatrixXd& V2, Eigen::MatrixXi& E2, Eigen::VectorXd& T2, Eigen::MatrixXi& B2, double minLength);
};
