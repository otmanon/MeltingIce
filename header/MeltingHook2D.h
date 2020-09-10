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
#include <igl/triangle/triangulate.h>

struct ThermalObject
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
	Eigen::VectorXi interfaceIndices; // indices of vertices on the interface
	Eigen::VectorXi boundaryIndices; // indices of vertices at mesh boundary (for solid it's the same as interfaceIndices, for liquid, it includes interface Indeces as well as border

	Eigen::MatrixXd EM; //Edge Midpoints
	Eigen::MatrixXd EN; //Edge Normals
	Eigen::MatrixXd VN; //2D normal stored at each vertex by averadging incident edge normals

	/*
	Diffuses heat by solving the heat equation by minmizing a semi-implicit time update of the Energy Functional :
	(u_n+1)^T (I - dt * C)u_n+1 -  (u_n+1)^T(u_n)

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
		Eigen::SparseMatrix<double> C, I, Q; //cotan matrix, identity matrix, quadratic coefficient term
		Eigen::VectorXd B = Eigen::VectorXd::Zero(V.rows(), 1); // linear coefficients
		B = -T; //Linear coefficients are equal to last timesteps solved Temperature field

		igl::cotmatrix(V, F, C);
		I.resize(V.rows(), V.rows());
		I.setIdentity();
		Q = I - dt * C;
		igl::min_quad_with_fixed_data<double> mqwf;
		Eigen::VectorXd Beq;
		Eigen::SparseMatrix<double> Aeq;
		igl::min_quad_with_fixed_precompute(Q, boundaryIndices, Aeq, false, mqwf);
		igl::min_quad_with_fixed_solve(mqwf, B, Tb, Beq, T);

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
			igl::readOFF(modelFilepath, solid.V, solid.F);

		solid.V *= 2;

		retriangulateSolidMesh();
		initSolidTemperatures();

		createLiquidMesh();
		initLiquidTemperatures();
		
		//Assemble a "combined vectors for visualisation"
		allV.resize(solid.V.rows() + liquid.V.rows(), solid.V.cols());
		

		allF.resize(solid.F.rows() + liquid.F.rows(), solid.F.cols());
		

		allT.resize(solid.V.rows() + liquid.V.rows(), 1);

		updateGlobalFields();

		solid.VGrad = Eigen::MatrixXd::Zero(solid.V.rows(), solid.V.cols());
		liquid.VGrad = Eigen::MatrixXd::Zero(liquid.V.rows(), liquid.V.cols());

		origV = allV;
		origF = allF;

		
	}

	/*
	Updates global fields allV, allF and allT by concatenating them into one array
	*/
	void updateGlobalFields()
	{
		allV << solid.V, liquid.V;
		allF << solid.F, liquid.F.array() + solid.V.rows();
		allT << solid.T, liquid.T;

	}

	/*
	ImGui render... not sure how this works
	*/
	virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
	{
		ImGui::Checkbox("stepping mode", &stepping_mode);
		ImGui::Checkbox("step sim", &stepping_flag);

		
		ImGui::Checkbox("render Fluid", &renderFluid);
		ImGui::Checkbox("render Solid", &renderSolid);


		ImGui::Checkbox("do diffusion", &diffusion_flag);
		ImGui::Checkbox("do melting", &melting_flag);
		ImGui::SliderFloat("dt", &dt, 1e-6, 1e1, "%.5f", 10.0f);
		ImGui::SliderFloat("phi", &phi, 0, 100);
		ImGui::SliderFloat("vis scale", &vis_scale, 1e-4, 1e-1, "%.8f", 10.0f);
	}

	/*
	Generates liquid mesh by wrapping bounding box around solid. mesh, scaling box by 2
	*/
	void createLiquidMesh()
	{

		//make mesh for "liquid/air" which will surround our object.
		Eigen::RowVector3d max = solid.V.colwise().maxCoeff();
		Eigen::RowVector3d min = solid.V.colwise().minCoeff();
		int h = max(1) - min(1);
		int w = max(0) - min(0);
		Eigen::MatrixXd canvasBorderV;
		canvasBorderV.resize(4, 3);
		canvasBorderV <<	max(0) + w * 0.5f, max(1) + h * 0.5f, 0.0f,   
							max(0) + w * 0.5f, min(1) - h * 0.5f, 0.0f,  
							min(0) - w * 0.5f, min(1) - h * 0.5f, 0.0f,   
							min(0) - w * 0.5f, max(1) + h * 0.5f, 0.0f;

	//	canvasBorderV *= 1.5f;
		int numInterfaceIndices = solid.interfaceIndices.rows();
		liquid.V = Eigen::MatrixXd::Zero(numInterfaceIndices + 4, 3);
		int maxInterfaceVertexIndex = solid.interfaceIndices.colwise().maxCoeff()(0);
		Eigen::VectorXi solidIndex2LiquidIndex = Eigen::VectorXi::Constant(maxInterfaceVertexIndex + 1, -1); //maps from boundary vertices in solid mesh, to boundary vertices in liquid mesh
		
																										 //boundary vertices in liquid mesh will be first and in order. 
		//the index of each element in the vector will come from the index of the boundary node in the solid mesh
		//the value at each entry in our vector is -1 if the index is not a boundary index of the solid mesh, or it is
		// equal to i, the corresponding index in the liquid mesh

		liquid.E = Eigen::MatrixXi::Zero(solid.E.rows() + 4, solid.E.cols());
		
		for (int i = 0; i < solid.interfaceIndices.rows() + 4; i++)
		{
			if (i < numInterfaceIndices)
			{
				liquid.V.row(i) = solid.V.row(solid.interfaceIndices(i));
				solidIndex2LiquidIndex(solid.interfaceIndices(i)) = i;
			}
			else
			{
				liquid.V.row(i) = canvasBorderV.row(i - numInterfaceIndices);
			}
		}

		int solidIndex1, solidIndex2, liquidIndex1, liquidIndex2;
		
		for (int i = 0; i < solid.E.rows(); i++)
		{
			solidIndex1 = solid.E(i, 0);
			solidIndex2 = solid.E(i, 1);

			//map from solid 2 liquid boundary index
			liquidIndex1 = solidIndex2LiquidIndex(solidIndex1);
			liquidIndex2 = solidIndex2LiquidIndex(solidIndex2);

			liquid.E(i, 0) = liquidIndex1;
			liquid.E(i, 1) = liquidIndex2;
		}

		// add the last few edges for the bounding box
		liquid.E(solid.E.rows() + 0, 0) = numInterfaceIndices + 0;	//top right Vert to bottom right
		liquid.E(solid.E.rows() + 0, 1) = numInterfaceIndices + 1;  
		liquid.E(solid.E.rows() + 1, 0) = numInterfaceIndices + 1;	//bottom right to bottom left
		liquid.E(solid.E.rows() + 1, 1) = numInterfaceIndices + 2;
		liquid.E(solid.E.rows() + 2, 0) = numInterfaceIndices + 2;	//bottom left to top left
		liquid.E(solid.E.rows() + 2, 1) = numInterfaceIndices + 3;
		liquid.E(solid.E.rows() + 3, 0) = numInterfaceIndices + 3;	//top left to top right
		liquid.E(solid.E.rows() + 3, 1) = numInterfaceIndices + 0;


		//get vertex index that is INSIDE of solid
		int interiorIndex = -1;
		Eigen::Vector3d interiorVertex;
		for (int i= 0; i < solidIndex2LiquidIndex.rows(); i++)
		{
			if (solidIndex2LiquidIndex(i) == -1)
			{
				interiorIndex = i;
				interiorVertex = solid.V.row(i);
			}
		}
		Eigen::MatrixXd liquidV2D;
		Eigen::MatrixXd H;
		H.resize(1, 2);
		H << interiorVertex(0), interiorVertex(1);
		
		convert3DVerticesTo2D(liquid.V, liquidV2D);
		//convert liquidV to 2D
		Eigen::MatrixXd V2, F2;
		igl::triangle::triangulate(liquidV2D, liquid.E, H, "a0.005q", V2, liquid.F);
		convert2DVerticesTo3D(V2, liquid.V);
	
	}

	/*
	Retriangulates mesh, by taking boundary edges/vertices, and filling out the interior
	*/
	void retriangulateSolidMesh()
	{
		// Find boundary edges
		
		igl::boundary_facets(solid.F, solid.E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.

		// Find boundary vertices
		Eigen::VectorXi IA, IC;  //gets indices of vertices on boundary and puts them in b
		igl::unique(solid.E, solid.interfaceIndices, IA, IC);

		Eigen::MatrixXd H;
		Eigen::MatrixXd V2;
		Eigen::MatrixXi F2;


		Eigen::MatrixXd V2D = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

		//Fill 2D V
		for (int i = 0; i < V2D.rows(); i++)
		{
			V2D(i, 0) = solid.V(i, 0);
			V2D(i, 1) = solid.V(i, 1);
		}

		igl::triangle::triangulate(V2D, solid.E, H, "a0.005q", V2, F2);
		solid.V = Eigen::MatrixXd::Zero(V2.rows(), 3);
		for (int i = 0; i < solid.V.rows(); i++)
		{
			solid.V(i, 0) = V2(i, 0);
			solid.V(i, 1) = V2(i, 1);
		}
		solid.F = F2;

	
		//We Now have NEW boundary vertices
		igl::boundary_facets(solid.F, solid.E);
		igl::unique(solid.E, solid.interfaceIndices, IA, IC);
	}

	/*
	Initializes the mesh vertices temperatures... 0 on border, -10 on interior.
	Also fills out Tb, the boundary temperature vector.
	*/
	void initSolidTemperatures()
	{
		//T.resize(V.rows()); // each vertex has temperature of -10
		solid.T = Eigen::VectorXd::Constant(solid.V.rows(), 1, 0.0);
	
		//Fills boundary temperature vertices. Will be useful for final solve
		solid.Tb = Eigen::VectorXd::Constant(solid.interfaceIndices.rows(), solid.interfaceIndices.cols(), 0.0f); //boundary temperature always held at freezing
		
		//Set boundary vertices to precet temperature of 10C
		igl::slice_into(solid.Tb, solid.interfaceIndices, 1, solid.T);
	}

	/*
	Initializes the liquid mesh vertices' tempratures. 10 on the canvas exterior, 0 on the interior. 
	Also fills out Tb, and finds boundary indices for the liquid mesh. 
	*/
	void initLiquidTemperatures()
	{
		float epsilon = 1e-6;
		float maxX, maxY, minX, minY; //coordinates of "bounding box" of liquid mesj
		liquid.T = Eigen::VectorXd::Constant(liquid.V.rows(), 1, 0);

		maxX = liquid.V.colwise().maxCoeff()(0);
		maxY = liquid.V.colwise().maxCoeff()(1);
		minX = liquid.V.colwise().minCoeff()(0);
		minY = liquid.V.colwise().minCoeff()(1); //this info will be useful when differentiating boundary edge on interface, to boundary edge on bounding box.

		Eigen::VectorXi IA, IC;  //gets indices of vertices on boundary and puts them in b
		igl::boundary_facets(liquid.F, liquid.E);
		igl::unique(liquid.E, liquid.boundaryIndices, IA, IC);
		
		liquid.interfaceIndices.resize(liquid.boundaryIndices.rows(), solid.interfaceIndices.cols());
		liquid.Tb.resize(liquid.boundaryIndices.rows());

		auto onBoundary = [maxX, maxY, minX, minY, epsilon](Eigen::Vector3d v1)->bool 
		{
			if (v1.x() - maxX > -epsilon|| v1.x() - minX < epsilon || v1.y() - maxY > -epsilon || v1.y() - minY < epsilon)
			{
				return true;
			}
			else
				return false;
		};

		//
		// If both vertices are not bounding box vertices, then they are interface indices
		float x1, x2, y1, y2;
		Eigen::Vector3d v1, v2;
		int interfaceIndex = 0;
		for (int i = 0; i < liquid.boundaryIndices.rows(); i++)
		{
			v1 = liquid.V.row(liquid.boundaryIndices(i));
			//get first index. If one of them is on boundary, both of them will be.
			if (onBoundary(v1))
			{ //bounding box kept at 10 on its edges
				liquid.Tb(i) = 100;
			}
			else //solid/liquid interface kept at 0... for now!
			{
				liquid.Tb(i) = 0;
				liquid.interfaceIndices(interfaceIndex) = liquid.boundaryIndices(i);
				interfaceIndex++;
			

			}
		}
		liquid.interfaceIndices.conservativeResize(interfaceIndex, liquid.boundaryIndices.cols());
		//Set boundary vertices to precet temperature of 10C
		igl::slice_into(liquid.Tb, liquid.boundaryIndices, 1, liquid.T);
	}
	
	/*
	Given gradient of scalar field accross each vertex, calculates the gradient at each vertex (average from neighborin gfaces)
	*/
	void calculateVertexGradientSolid(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& FGrad)
	{
		solid.VGrad = Eigen::MatrixXd::Zero(V.rows(), 3);
		Eigen::VectorXd VGradWeight = Eigen::VectorXd::Zero(V.rows()); //keeps track of number of faces that have contributed to vertex
		for (int i = 0; i < FGrad.rows(); i++) //loop through faces, distribute gradients to each vertex
		{
			int v1_index = F(i, 0), v2_index = F(i, 1), v3_index = F(i, 2);

			solid.VGrad.row(v1_index) += FGrad.row(i);
			VGradWeight(v1_index) += 1;

			solid.VGrad.row(v2_index) += FGrad.row(i);
			VGradWeight(v2_index) += 1;

			solid.VGrad.row(v3_index) += FGrad.row(i);
			VGradWeight(v3_index) += 1;

		}

		for (int i = 0; i < solid.VGrad.rows(); i++)
		{

			solid.VGrad.row(i) /= VGradWeight(i);
		}

	}

	/*
	Calculates the gradient of the temperature field near the surface. Assumes gradient is constant on the outside.
	Then, moves the surface inwards based on the stefan condition:
	Lv = ks dTs/dn - kl dTl/dn
	*/
	void meltSurface()
	{

		Eigen::MatrixXd DTDN = Eigen::MatrixXd::Zero(solid.VN.rows(), solid.VN.cols());
		// For each vertex, get the normal and the gradient, then dot the normal against the gradient
		for (int i = 0; i < solid.V.rows(); i++)
		{
			Eigen::Vector3d normal = solid.VN.row(i);
			Eigen::Vector3d grad = - ( liquid.VGrad.row(i) + solid.VGrad.row(i));
			Eigen::Vector3d dTdn = normal.dot(grad) * normal;
			DTDN.row(i) = dTdn;
		}
		solid.V = solid.V - DTDN * dt;
	}

	/*
	Calculates all info having to do with gradient of temperature field
	*/
	void calculateGradientInfo()
	{
		Eigen::SparseMatrix<double> G;  //Gradient operator. Takes scalar (T) stored at each vertex, returns gradient of T at each face.
		igl::grad(solid.V, solid.F, G);	//Get gradient operator

		solid.FGrad = Eigen::Map<const Eigen::MatrixXd>((G*solid.T).eval().data(), solid.F.rows(), 3); //Gradient of T on each face.

		calculateVertexGradientSolid(solid.V, solid.F, solid.FGrad); //Go through each face and distribute gradient to face's vertices.


		calculateBoundaryMidpointsAndNormals();
		//Eigen::MatrixXd VGradExt = getVertexGradientExt(V, F, VN);
		calculateVertexGradientLiquid();
	}

	/*
	Loops through all boundary edges. Deposits their contributions to their boundary vertices. 
	*/
	void calculateVertexGradientLiquid()
	{
		Eigen::VectorXi bndEdges;
		igl::boundary_loop(solid.F, bndEdges);
		//Assuming boundary is closed loop, each vertex has 2 edges. 
		liquid.VGrad = Eigen::MatrixXd::Zero(solid.V.rows(), 3);
		
		for (int i = 0; i < bndEdges.rows(); i++) //loop through faces, distribute gradients to each vertex
		{
			int i1 = i;
			int i2 = (i + 1) % bndEdges.rows(); //Mod operators bc last index is attached to first index

			liquid.VGrad.row(bndEdges(i1)) += -0.5f *  phi * solid.VN.row(bndEdges(i1));
			liquid.VGrad.row(bndEdges(i2)) += -0.5f * phi * solid.VN.row(bndEdges(i2));

		}
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
		igl::boundary_loop(solid.F, bnd);
		solid.EM.resize(bnd.rows(), 3); //M holds the midpoints for each boundary edge. x, y, and z coords
		solid.EN.resize(bnd.rows(), 3); //N holds the normals for each boundary edge. x, y, and z coords
		solid.VN.resize(solid.V.rows(), 3); //VN holds the normals for each vertex
		solid.VN.setZero();

		for (int i = 0; i < bnd.rows(); i++)
		{
			int i1 = i;
			int i2 = (i + 1) % bnd.rows(); //Mod operators bc last index is attached to first index

			Eigen::Vector3d midpoint = (solid.V.row(bnd(i2)) + solid.V.row(bnd(i1))) / 2;
			Eigen::Vector3d r = (solid.V.row(bnd(i2)) - solid.V.row(bnd(i1)));
			Eigen::Vector3d z(0, 0, 1);
			Eigen::Vector3d normal = r.cross(z);
			normal.normalize();
			solid.EM.row(i) = midpoint;
			solid.EN.row(i) = normal;

			//Now deposit the edge normal towards the vertex normal
			solid.VN.row(bnd(i1)) += solid.EN.row(i);
			solid.VN.row(bnd(i2)) += solid.EN.row(i);

		}

		for (int i = 0; i < solid.VN.rows(); i++)
		{
			solid.VN.row(i).normalize();
		}
	};

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
					liquid.diffuseHeat(dt);
					calculateGradientInfo();
					updateGlobalFields();
				}
				if(melting_flag)
				//	meltSurface();
				stepping_flag = false;
			}
		}
		else
		{
			liquid.diffuseHeat(dt);
			calculateGradientInfo();
			updateGlobalFields();
			//meltSurface();
		}

		return false;
	}


	/*

	*/
	virtual void updateRenderGeometry()
	{
		if (renderFluid)
		{
			renderV = liquid.V;
			renderF = liquid.F;
			renderT = liquid.T;
		}
		else if (renderSolid)
		{
			renderV = solid.V;
			renderF = solid.F;
			renderT = solid.T;
		}
		else {
			renderV = allV;
			renderF = allF;
			renderT = allT;
		}


	}


	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer &viewer)
	{
		viewer.data().clear();
	//	viewer.data().clear_edges();
		viewer.data().set_mesh(renderV, renderF);
		viewer.data().set_data(renderT, 0, 100);

	//	const Eigen::RowVector3d black(0, 0, 0);
	//	viewer.data().add_edges(solid.V, solid.V + vis_scale * solid.VGrad, black);

	//	const Eigen::RowVector3d red(1, 0, 0);
	//	viewer.data().add_edges(solid.V, solid.V + vis_scale * solid.VGrad, red);
	}

private:
	bool renderFluid = true;
	bool renderSolid = false;

	bool stepping_mode = false;
	bool stepping_flag = false;
	bool melting_flag = true;
	bool diffusion_flag = true;
	float vis_scale = 0.0001;
	float phi = 50; //flux applied at each edge
	float dt = 1e-1; //timestep

	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd allV; // includes Solid geometry and liquid geometry.
	Eigen::MatrixXi allF; 
	Eigen::VectorXd allT;

	Eigen::MatrixXd renderV; // Mesh description that will be rendered (ie either solid geometry, liquid geometry, or both
	Eigen::MatrixXi renderF;
	Eigen::VectorXd renderT;

	std::string modelFilepath = "data/2dModels/elephant.off";

	ThermalObject solid;
	ThermalObject liquid;
	
};