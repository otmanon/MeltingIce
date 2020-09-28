#include "PhysicsHook.h"
#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <math.h>
#include <igl/slice_into.h>
#include <igl/unique.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <igl/grad.h>;
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/winding_number.h>
#include <igl/triangle/triangulate.h>
#include <igl/doublearea.h>

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
	Eigen::MatrixXi E; // Edges belonging to that domain. The indices in the edges refer to vertices in the parent Domains V matrix
	Eigen::VectorXd T; // Temperatures associated to the vertices in this interface
	Eigen::VectorXi Vi;// Indices from parentDomain's Vertices list that belongs to this domain

	Eigen::MatrixXd MidV;//contains midpoitns of all boundary edges
	Eigen::MatrixXd NormalsE;//contains normals at each boundary edge

};

struct Domain
{
	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd V;	//mesh geometry for solid
	Eigen::MatrixXi F;
	Eigen::MatrixXi E;  //Matrix of edges (N, 2), where each row is an edge, whose entries are both vertex indices
	
	Eigen::MatrixXd FGrad; //Gradient on faces 
	Eigen::MatrixXd VGrad; //Gradient on vertices coming from solid(average of nearby faces) 

	Eigen::VectorXd T; // Temperature at each vertex
	Eigen::VectorXd Tb;	// temperature at each boundary vertex


	Eigen::MatrixXd VN; //2D normal stored at each vertex by averadging incident edge normals
	Eigen::VectorXd W; //Winding number...

	Interface I;
	Interface Boundary;
	SubDomain S;
	SubDomain L;

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
	void diffuseHeat(float dt)
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

	/*
	Calculates the gradient of the temperature field near the surface.
	Then, moves the surface inwards based on the stefan condition:
	Lv = ks dTs/dn - kl dTl/dn
	*/
	void melt(float dt, float latentHeat)
	{
		calculateVertexGradient(); //Go through each face and distribute gradient to face's vertices.
		calculateBoundaryMidpointsAndNormals();

		Eigen::MatrixXd DTDN = Eigen::MatrixXd::Zero(VN.rows(), VN.cols());
		// For each vertex, get the normal and the gradient, then dot the normal against the gradient
		for (int i = 0; i < V.rows(); i++)
		{
			Eigen::Vector3d normal = VN.row(i);
			Eigen::Vector3d grad = (VGrad.row(i));
			Eigen::Vector3d dTdn = normal.dot(grad) * normal;
			if (!isBoundaryVertex(V.row(i))) //only melt boundary vertices
				DTDN.row(i) = dTdn;
		}
		V = V - DTDN * dt/latentHeat;
	}

	/*
	Get boundary vertices and normals
	Takes as input the vertices and faces of the mesh, V and F
	Outputs :
	EM, the modpoint of each boundary edge
	EN the normal at each boundary edge
	VN the normal at each boundary vertex
	*/
	void calculateBoundaryMidpointsAndNormals()
	{
		Eigen::VectorXi bnd;
		igl::boundary_loop(S.F, bnd);

		I.MidV.resize(bnd.rows(), 3); //M holds the midpoints for each boundary edge. x, y, and z coords
		I.NormalsE.resize(bnd.rows(), 3); //N holds the normals for each boundary edge. x, y, and z coords
		VN.resize(V.rows(), 3); //VN holds the normals for each vertex
		VN.setZero();

		for (int i = 0; i < bnd.rows(); i++) //loop through all boundary edges
		{
			int i1 = i;
			int i2 = (i + 1) % bnd.rows(); //Mod operators bc last index is attached to first index

			Eigen::Vector3d midpoint = (V.row(bnd(i2)) + V.row(bnd(i1))) / 2;
			Eigen::Vector3d r = (V.row(bnd(i2)) - V.row(bnd(i1)));
			Eigen::Vector3d z(0, 0, 1); 
			Eigen::Vector3d normal = r.cross(z);
			normal.normalize();
			I.MidV.row(i) = midpoint;
			I.NormalsE.row(i) = normal;

			//Now deposit the edge normal towards the vertex normal
			VN.row(bnd(i1)) += I.NormalsE.row(i);
			VN.row(bnd(i2)) += I.NormalsE.row(i);
		

		}
		for (int i = 0; i < VN.rows(); i++)
		{
				VN.row(i) /= 2;
		}

	};



	/*
	Given gradient of scalar field accross each vertex, calculates the gradient at each vertex (average from neighborin gfaces)
	*/
	void calculateVertexGradient()
	{
		Eigen::SparseMatrix<double> G;  //Gradient operator. Takes scalar (T) stored at each vertex, returns gradient of T at each face.
		igl::grad(V, F, G);	//Get gradient operator

		FGrad = Eigen::Map<const Eigen::MatrixXd>((G*T).eval().data(), F.rows(), 3); //Gradient of T on each face.

		VGrad = Eigen::MatrixXd::Zero(V.rows(), 3);

		
		Eigen::VectorXd VGradWeight = Eigen::VectorXd::Zero(V.rows()); //keeps track of number of faces that have contributed to vertex
		
		
		//get triangle areas
		Eigen::VectorXd A;
		igl::doublearea(V, F, A);
		A *= 0.5f;
		float tArea = 1/3.0f;
		//std::cout << A << std::endl;
		for (int i = 0; i < FGrad.rows(); i++) //loop through faces, distribute gradients to each vertex
		{
			int v1_index = F(i, 0), v2_index = F(i, 1), v3_index = F(i, 2);
			//tArea = A(i)/3.0f;
		
			VGrad.row(v1_index) += FGrad.row(i)*tArea;
			VGradWeight(v1_index) += 1;

			VGrad.row(v2_index) += FGrad.row(i)*tArea;
			VGradWeight(v2_index) += 1;

			VGrad.row(v3_index) += FGrad.row(i)*tArea;
			VGradWeight(v3_index) += 1;

		}



	}

	bool isBoundaryVertex(Eigen::Vector3d v)
	{
		return (v(0) == maxX || v(0) == minX || v(1) == maxY || v(1) == minY);
	
	}
};


class MeltingHook2D : public PhysicsHook
{
public:
	MeltingHook2D() : PhysicsHook() {}

	/*
	Initializes simulation. 
	*/
	virtual void initSimulation()
	{
	
		// Load a mesh in OFF format
		//if (modelFilepath.substr(modelFilepath.length() - 3) == ".obj")
		//	igl::readOBJ(modelFilepath, origV, origF);
	//	else if (modelFilepath.substr(modelFilepath.length() - 3) == ".off")
		Eigen::MatrixXd V;
		Eigen::MatrixXi F, E;
		igl::readOFF(modelFilepath, V, F);
		igl::boundary_facets(F, E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.


		initMesh.V *= 1;
		triangulateDomain(V, F, E);
		
		segmentDomain(V, E, true);
		
	}

	/*
	Divides the domain into solid/liquid/boundary
	Input:
		V: num_Verts x 3 matrix of vertex coordinates
		E: Edges representing boundary we want to perform inside/outside segmentation of
	*/
	void segmentDomain(Eigen::MatrixXd V, Eigen::MatrixXi E, bool init)
	{
		Eigen::VectorXd W;
		Eigen::MatrixXd BC;
		// Compute barycenters of all tets
		igl::barycenter(D.V, D.F , BC);
		igl::winding_number(V, E, BC, D.W); //winding number at baricenters

		//normalize
		D.W = (D.W.array() - D.W.minCoeff()) / (D.W.maxCoeff() - D.W.minCoeff());
		W = D.W;

		//Count how many vertices are inside Solid and outside Solid
		Eigen::MatrixXi solidF((W.array() > 0.9f).count(), 3 ); // faces inside solid
		Eigen::MatrixXi liquidF((W.array() < 0.9f).count(), 3); // faces inside solid
		Eigen::VectorXd T(D.V.rows());
		//update Solid domain and indices 
		int indexS = 0;
		int indexL = 0;
		for (int i = 0; i < D.F.rows(); i++)
		{
			if (D.W(i) > 0.9f) //this point is inside domain
			{
				solidF.row(indexS) = D.F.row(i);
				indexS++;
			}
			else
			{
				liquidF.row(indexL) = D.F.row(i);
				indexL++;
			}
		}

		Eigen::MatrixXi interfaceEdges;
		Eigen::VectorXi interiorIndices, interiorIndices2, allIndices, exteriorIndices, interfaceIndices, IA, IC;
		igl::unique(solidF, interiorIndices);											//interiorIndices contains index of interior vertices
		igl::colon(0, D.V.rows() - 1, allIndices);										//allIndices contains all indeces
		igl::setdiff(allIndices, interiorIndices, exteriorIndices, IA);					//exterior indices 
		igl::boundary_facets(solidF, interfaceEdges);									//get edges of solid
		igl::unique(interfaceEdges, interfaceIndices);									//get interface indices

		//separate  boundary indices from interior indices.
		igl::setdiff(interiorIndices, interfaceIndices, interiorIndices2, IA);
		interiorIndices = interiorIndices2;

		//set up interior/exterior Temperatures
		Eigen::VectorXd interiorT = Eigen::VectorXd::Constant(interiorIndices.rows(), 0.0f);
		Eigen::VectorXd interfaceT = Eigen::VectorXd::Constant(interfaceIndices.rows(), 0.0f);
		Eigen::VectorXd exteriorT = Eigen::VectorXd::Constant(exteriorIndices.rows(), 0.0f);

		//Set bounding box indeces and temperatures
		Eigen::VectorXd  boundaryT;
		igl::boundary_facets(D.F, D.Boundary.E);
		igl::unique(D.Boundary.E, D.Boundary.Vi);
		D.Boundary.T = Eigen::VectorXd::Constant(D.Boundary.Vi.rows(), 10);

		//Updates main T with subdomain Ts
		igl::slice_into(interiorT, interiorIndices , 1, T);
		igl::slice_into(exteriorT, exteriorIndices, 1, T);
		igl::slice_into(interfaceT, interfaceIndices, 1, T);
		igl::slice_into(D.Boundary.T, D.Boundary.Vi, T);



		//Fill out subdomain info
		D.S.F = solidF;
		if (init)
		{
			D.S.Vi = interiorIndices;
			D.S.T = interiorT;
		}
	
		D.L.F = liquidF;
		if (init)
		{
			D.L.Vi = exteriorIndices;
			D.L.T = exteriorT;
		}

		D.I.E = interfaceEdges;
		if (init)
		{
			D.I.Vi = interfaceIndices;

			D.I.T = interfaceT;
		
			D.T = T;
			D.VGrad = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
			D.VN = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
		}
	}

	/*
	Given solid mesh, extracts its boundary, wraps a bounding box around it, triangulates 
	entire domain in one sweep, while maintaining elephant boundary edges
	*/
	void triangulateDomain(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXi E)
	{

		Eigen::VectorXi bi;
		float avgLength = igl::avg_edge_length(V, F);
		// Find boundary vertices
		igl::boundary_facets(F, E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.
		Eigen::VectorXi IA, IC;  //gets indices of vertices on boundary and puts them in b
		igl::unique(E, bi, IA, IC);

		// create Surrounding Bounding Boxe
		Eigen::RowVector3d max = V.colwise().maxCoeff();
		Eigen::RowVector3d min = V.colwise().minCoeff();


		float h = max(1) - min(1);
		float w = max(0) - min(0);

		int NV = V.rows();
		int NE = E.rows();


		V.conservativeResize( V.rows() + 4, 2);
		E.conservativeResize(E.rows() + 4, E.cols());

		D.maxX = max(0) + w * 0.5f;
		D.maxY = max(1) + h * 0.5f;
		D.minX = min(0) - w * 0.5f;
		D.minY = min(1) - h * 0.5f;
		//Add new vertices
		V.row(NV + 0) = Eigen::Vector2d(D.maxX, D.maxY);
		V.row(NV + 1) = Eigen::Vector2d(D.maxX, D.minY);
		V.row(NV + 2) = Eigen::Vector2d(D.minX, D.minY);
		V.row(NV + 3) = Eigen::Vector2d(D.minX, D.maxY);

		// add the last few edges for the bounding box
		E(NE + 0, 0) = NV + 0;	//top right Vert to bottom right
		E(NE + 0, 1) = NV + 1;
		E(NE + 1, 0) = NV + 1;	//bottom right to bottom left
		E(NE + 1, 1) = NV + 2;
		E(NE + 2, 0) = NV + 2;	//bottom left to top left
		E(NE + 2, 1) = NV + 3;
		E(NE + 3, 0) = NV + 3;	//top left to top right
		E(NE + 3, 1) = NV + 0;


		//horizontal edges(easy)
		Eigen::MatrixXd V2D, V2;
		Eigen::MatrixXi F2;
	//	convert3DVerticesTo2D(V, V2D);
		Eigen::MatrixXd H;
		igl::triangle::triangulate(V, E, H, "a0.005q", V2, F2);
		convert2DVerticesTo3D(V2, V);
		D.V = V;
		D.F = F2;
		allV = D.V;
		D.F;
		
	
	}

	/*
	Updates domain. When we melt, small triangles near the interface can occur. Need to handle those by either deleting unnecessary vertices
	or coming up with new triangle configuration.
	*/
	void updateDomain()
	{
		//Save previous version of mesh
		Eigen::MatrixXd V = D.V;
		Eigen::MatrixXi F = D.F;

		//re triangulate domain
		retriangulateDomain();
		//re segment domain
		segmentDomain(V, D.I.E, false);
	
	}


	void retriangulateDomain()
	{

		Eigen::MatrixXi boundaryEdges;

		boundaryEdges.resize(D.Boundary.E.rows() + D.I.E.rows(), D.I.E.cols());
		boundaryEdges << D.Boundary.E, D.I.E;
		//horizontal edges(easy)
		Eigen::MatrixXd V2D, V2;
		Eigen::MatrixXi F2;
		convert3DVerticesTo2D(D.V, V2D);
		Eigen::MatrixXd H;
		igl::triangle::triangulate(V2D, boundaryEdges, H, "-yy", V2, F2);
		convert2DVerticesTo3D(V2, D.V);
		D.F = F2;
	//	D.VGrad = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
	//	D.VN = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
	}

	/*
	ImGui render... not sure how this works
	*/
	virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
	{
		ImGui::Checkbox("stepping mode", &stepping_mode);
		ImGui::Checkbox("step sim", &stepping_flag);

		ImGui::CollapsingHeader("Display");
		ImGui::Checkbox("render Fluid", &renderFluid);
		ImGui::Checkbox("render Solid", &renderSolid);
		ImGui::Checkbox("render Normals", &renderNormals);
		ImGui::Checkbox("render Temp Gradient", &renderTGrad);


		ImGui::Checkbox("do diffusion", &diffusion_flag);
		ImGui::Checkbox("do melting", &melting_flag);
		ImGui::Checkbox("re-triangulate", &retriangulate);
		ImGui::SliderFloat("dt", &dt, 1e-6, 1e1, "%.5f", 10.0f);
		ImGui::SliderFloat("phi", &phi, 0, 100);
		ImGui::SliderFloat("vis scale", &vis_scale, 1e-4, 1e-1, "%.8f", 10.0f);
	}

	/*
	Converts 3D vertices (used in libigl) to 2D (used in triangle)
	*/
	void convert3DVerticesTo2D(Eigen::MatrixXd& v3D, Eigen::MatrixXd& v2D)
	{
		v2D.resize(v3D.rows(), 2);// = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

		//Fill 2D V
		for (int i = 0; i < v2D.rows(); i++)
		{
			v2D(i, 0) = v3D(i, 0);
			v2D(i, 1) = v3D(i, 1);
		}
	}

	/*
	converts 2D vertices to 3D 
	*/
	void convert2DVerticesTo3D( Eigen::MatrixXd& v2D, Eigen::MatrixXd& v3D)
	{
		v3D.resize(v2D.rows(), 3);// = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

		//Fill 2D V
		for (int i = 0; i < v3D.rows(); i++)
		{
			v3D(i, 0) = v2D(i, 0);
			v3D(i, 1) = v2D(i, 1);
			v3D(i, 2) = 0;
		}
	}

	/*
	Simulates one timestep
	*/
	virtual bool simulateOneStep()
	{
		if (stepping_mode)
		{
			if (stepping_flag)
			{
				if (diffusion_flag)
				{
					D.diffuseHeat(dt);
				}
				if(melting_flag)
					D.melt(dt, 10);
				stepping_flag = false;
			}
		}
		else
		{
			if (diffusion_flag)
			{
				D.diffuseHeat(dt);
			}
			if (melting_flag)
			{
				D.melt(dt, 10);
			}
			if (retriangulate)
			{
				updateDomain();//once we melt we need to make sure our geometry is still nice. 
			}
				
			//calculateGradientInfo();
			//meltSurface();
		}

		return false;
	}


	/*

	*/
	virtual void updateRenderGeometry()
	{
		//V and T always the same
		renderV = D.V;
		renderT = D.T;

		if (renderFluid && !renderSolid)
			renderF = D.L.F;
		else if (renderSolid && !renderFluid)
			renderF = D.S.F;
		else 
		{
			renderF = D.F;
		}
	}


	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer &viewer)
	{
	//	viewer.data().clear_edges();
		viewer.data().clear();
	//	viewer.data().clear_edges();
		viewer.data().set_mesh(renderV, renderF);
		viewer.data().set_data(renderT, 0, 10);

		if (renderTGrad)
		{
			const Eigen::RowVector3d green(0, 1, 0);
			viewer.data().add_edges(D.V, D.V + vis_scale * D.VGrad, green);

		}
		if (renderNormals)
		{
			const Eigen::RowVector3d red(1, 0, 0);
			viewer.data().add_edges(D.V, D.V + vis_scale * D.VN, red);
		}
	}

private:
	bool renderFluid = true;
	bool renderSolid = true;
	bool renderNormals = false;
	bool renderTGrad = false;

	
	bool stepping_mode = false;
	bool stepping_flag = false;
	bool melting_flag = true;
	bool diffusion_flag = true;
	bool retriangulate = true;

	float vis_scale = 0.01;
	float phi = 50; //flux applied at each edge
	float dt = 1e-2; //timestep

	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd allV; // includes Solid geometry and liquid geometry.
	Eigen::MatrixXi allF; 
	Eigen::VectorXd allT;

	Eigen::MatrixXd renderV; // Mesh description that will be rendered (ie either solid geometry, liquid geometry, or both
	Eigen::MatrixXi renderF;
	Eigen::VectorXd renderT;

	std::string modelFilepath = "data/2dModels/elephant.off";

	Domain initMesh;
	Domain liquid;

	Domain D;
};