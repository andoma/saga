#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include <stdio.h>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include "ext/imnodes/ImNodes.h"


#include "saga.h"

#include <condition_variable>
#include <vector>
#include <functional>


namespace ImNodes {
extern CanvasState* gCanvas;
}

namespace saga {

class UIWindow;

class IMUI : public UI {
public:

  ~IMUI()
  {
  }

  std::condition_variable work_cond_;
  std::mutex work_mutex_;
  std::vector<std::function<void(void)>> work_;

  std::vector<std::shared_ptr<UIWindow>> windows_;

  void run() override;

  void showGraph(const Graph& g, std::shared_ptr<Program> p) override;

  void dispatch(std::function<void(void)> work) {
    work_mutex_.lock();
    work_.push_back(work);
    work_cond_.notify_one();
    work_mutex_.unlock();
  }

  void handle() {
    work_mutex_.lock();
    auto work = std::move(work_);
    work_mutex_.unlock();

    for(auto &pow : work) {
      pow();
    }
  }
};


class UIWindow {
public:
  virtual ~UIWindow() {};
  virtual bool draw(IMUI &ui) = 0;
};







class UITensor : public UIWindow {
public:
  UITensor(std::shared_ptr<Tensor> t)
    : t_(t)
  {}

  ~UITensor()
  {
    if(texs_.size()) {
      printf("Deleted %zd textures\n", texs_.size());
      glDeleteTextures(texs_.size(), &texs_[0]);
    }
  }

  int zoom_ = 1;
  const std::shared_ptr<Tensor> t_;
  std::vector<unsigned int> texs_;
  bool draw(IMUI &ui);
};


bool
UITensor::draw(IMUI &ui)
{
  auto info = t_->info();
  bool open = true;
  if(ImGui::Begin(info.c_str(), &open)) {

    auto rgb = t_->toRGB();
    if(rgb) {
      ImGui::SliderInt("Zoom", &zoom_, 1, 8);
      auto ta = rgb->access();

      const int n = rgb->dims_[0];
      const int c = rgb->dims_[1];
      const int h = rgb->dims_[2];
      const int w = rgb->dims_[3];
      const size_t num_images = n * c;

      if(texs_.size() != num_images) {
        glDeleteTextures(texs_.size(), &texs_[0]);
        texs_.resize(num_images);
        glGenTextures(num_images, &texs_[0]);

        for(auto tex : texs_) {
          glBindTexture(GL_TEXTURE_2D, tex);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        }
      }

      const uint8_t *pixels = (const uint8_t *)ta->data();
      ImVec2 size(w * zoom_ , h * zoom_);
      Dims strides = ta->strides();

      int i = 0;
      for(int y = 0; y < n; y++) {
        for(int x = 0; x < c; x++) {
          glBindTexture(GL_TEXTURE_2D, texs_[i]);
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                       pixels + y * strides[0] + x * strides[1]);
          ImGui::Image((void *)(intptr_t)texs_[i], size);
          i++;
          if(x < c - 1)
            ImGui::SameLine();
        }
      }
    } else {

    }
  }
  ImGui::End();
  return !open;
}



static void
show_tensor(IMUI &ui, std::shared_ptr<Tensor> t,
            std::shared_ptr<Program> p)
{
  if(p)
    t = p->resolveTensor(t);
  ui.windows_.push_back(std::make_shared<UITensor>(t));
}




bool BeginNode(void* node_id, const char* title, ImVec2* pos, bool* selected)
{
    bool result = ImNodes::BeginNode(node_id, pos, selected);
    auto* storage = ImGui::GetStateStorage();

    float node_width = storage->GetFloat(ImGui::GetID("node-width"));

    if(ImGui::Button(title, ImVec2(node_width, 0.0)))
      ImGui::OpenPopup("my_select_popup");

    if(ImGui::BeginPopup("my_select_popup")) {
      ImGui::Button("Tensor");
      ImGui::Separator();
      ImGui::Button("Clear");
      ImGui::EndPopup();
    }


    ImGui::BeginGroup();
    return result;
}

void EndNode()
{
    // Store node width which is needed for centering title.
    auto* storage = ImGui::GetStateStorage();
    ImGui::EndGroup();
    storage->SetFloat(ImGui::GetID("node-width"), ImGui::GetItemRectSize().x);
    ImNodes::EndNode();
}

bool Slot(const char* title, int kind,
          std::shared_ptr<Node> node,
          IMUI &ui,
          std::shared_ptr<Program> p)
{
  using namespace ImNodes;

  auto* storage = ImGui::GetStateStorage();
  const auto& style = ImGui::GetStyle();
  const float CIRCLE_RADIUS = 5.f * gCanvas->zoom;
  ImVec2 title_size = ImGui::CalcTextSize(title);
  // Pull entire slot a little bit out of the edge so that curves connect into int without visible seams
  float item_offset_x = style.ItemSpacing.x * gCanvas->zoom;

  if(!ImNodes::IsOutputSlotKind(kind))
    item_offset_x = -item_offset_x;

  ImGui::SetCursorScreenPos(ImGui::GetCursorScreenPos() + ImVec2{item_offset_x, 0});

  if(ImNodes::BeginSlot(title, kind)) {

    auto* draw_lists = ImGui::GetWindowDrawList();

    // Slot appearance can be altered depending on curve hovering state.
    bool is_active = ImNodes::IsSlotCurveHovered() ||
      (ImNodes::IsConnectingCompatibleSlot() /*&& !IsAlreadyConnectedWithPendingConnection(title, kind)*/);

    ImColor color = gCanvas->colors[is_active ? ImNodes::ColConnectionActive : ImNodes::ColConnection];

    //    ImGui::PushStyleColor(ImGuiCol_Text, color.Value);

    if(ImNodes::IsOutputSlotKind(kind)) {
      // Align output slots to the right edge of the node.
      ImGuiID max_width_id = ImGui::GetID("output-max-title-width");
      float output_max_title_width = ImMax(storage->GetFloat(max_width_id, title_size.x), title_size.x);
      storage->SetFloat(max_width_id, output_max_title_width);
      float offset = (output_max_title_width + style.ItemSpacing.x) - title_size.x;
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);

      if(ImGui::SmallButton(title)) {
        show_tensor(ui, node->outputs_[title], p);
      }
      ImGui::SameLine();
    }

      ImRect circle_rect{
        ImGui::GetCursorScreenPos(),
          ImGui::GetCursorScreenPos() + ImVec2{CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2}
      };
      // Vertical-align circle in the middle of the line.
      float circle_offset_y = title_size.y / 2.f - CIRCLE_RADIUS;
      circle_rect.Min.y += circle_offset_y;
      circle_rect.Max.y += circle_offset_y;
      draw_lists->AddCircleFilled(circle_rect.GetCenter(), CIRCLE_RADIUS, color);

      ImGui::ItemSize(circle_rect.GetSize());
      ImGui::ItemAdd(circle_rect, ImGui::GetID(title));

      if(ImNodes::IsInputSlotKind(kind)) {
        ImGui::SameLine();
        if(ImGui::SmallButton(title)) {
          show_tensor(ui, node->inputs_[title], p);
        }
      }

      //      ImGui::PopStyleColor();
      ImNodes::EndSlot();

      // A dirty trick to place output slot circle on the border.
      ImGui::GetCurrentWindow()->DC.CursorMaxPos.x -= item_offset_x;
      return true;
    }
  return false;
}


class UINode;

class UIConnection {
public:
  UIConnection(const std::string &src_slot,
               std::shared_ptr<UINode> dst_node, const std::string &dst_slot)
    : src_slot_(src_slot)
    , dst_node_(dst_node)
    , dst_slot_(dst_slot)
  {}

  const std::string src_slot_;
  const std::shared_ptr<UINode> dst_node_;
  const std::string dst_slot_;
};



class UINode {
public:
  UINode(std::shared_ptr<Node> node, const ImVec2 &pos)
    : node_(node)
    , selected_(false)
    , pos_(pos)
  {}

  std::shared_ptr<Node> node_;
  bool selected_;
  ImVec2 pos_;

  std::vector<std::shared_ptr<UIConnection>> connections_;

};


class UIGraph : public UIWindow {
public:
  std::vector<std::shared_ptr<UINode>> nodes_;

  std::shared_ptr<Program> program_;

  bool draw(IMUI &ui);
  std::unique_ptr<ImNodes::CanvasState> state_;
};





void
IMUI::showGraph(const Graph& g, std::shared_ptr<Program> p)
{
  auto mappings = g.tensorMappings();
  auto uig = std::make_shared<UIGraph>();
  uig->program_ = p;

  std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<UINode>> nodemap;

  ImVec2 pos(100,100);
  for(auto &n : g.nodes_) {

    ImVec2 offset(0,0);

    if(n->inputs_.getv("x").size() > 1)
      offset.y += 100;

    auto y = n->outputs_["y"];
    if(mappings.first[y].size() > 1)
      offset.y += 200;


    auto uin = std::make_shared<UINode>(n, ImVec2(pos.x + offset.x,
                                                  pos.y + offset.y));

    uig->nodes_.push_back(uin);
    nodemap[n] = uin;
    pos.x += 150;

  }

  for(auto &n : g.nodes_) {
    for(auto &output : n->outputs_) {
      for(auto &input : mappings.first[output.second]) {
        auto uin = nodemap[n];
        auto c = std::make_shared<UIConnection>(output.first,
                                                nodemap[input.second],
                                                input.first);
        uin->connections_.push_back(c);
      }
    }
  }

  dispatch([=] {
    windows_.push_back(uig);
  });

}

bool
UIGraph::draw(IMUI &ui)
{
  bool open = true;

  if(ImGui::Begin("Graph", &open,
                  ImGuiWindowFlags_NoScrollbar |
                  ImGuiWindowFlags_NoScrollWithMouse)) {

    const auto& style = ImGui::GetStyle();

    if(!state_) {
      state_ = std::make_unique<ImNodes::CanvasState>();
      state_->style.curve_thickness = 2;
    }

    ImNodes::BeginCanvas(state_.get());
    for(auto &n : nodes_) {

      if(BeginNode(n.get(),
                   n->node_->type_.c_str(),
                   &n->pos_,
                   &n->selected_)) {



        // Render input slots
        ImGui::BeginGroup();
        for(auto &i : n->node_->inputs_) {
          Slot(i.first.c_str(), ImNodes::InputSlotKind(1), n->node_, ui,
               program_);
        }
        ImGui::EndGroup();

        // Move cursor to the next column
        ImGui::SetCursorScreenPos({ImGui::GetItemRectMax().x + style.ItemSpacing.x, ImGui::GetItemRectMin().y});

        // Begin region for node content
        ImGui::BeginGroup();


        // End region of node content
        ImGui::EndGroup();

        // Render output slots in the next column
        ImGui::SetCursorScreenPos({ImGui::GetItemRectMax().x + style.ItemSpacing.x, ImGui::GetItemRectMin().y});
        ImGui::BeginGroup();
        for(auto &o : n->node_->outputs_) {
          Slot(o.first.c_str(), ImNodes::OutputSlotKind(1), n->node_, ui,
               program_);
        }
        ImGui::EndGroup();

        for(auto &c : n->connections_) {
          ImNodes::Connection(c->dst_node_.get(),
                              c->dst_slot_.c_str(),
                              n.get(),
                              c->src_slot_.c_str());
        }
      }
      EndNode();
    }
    ImNodes::EndCanvas();
  }
  ImGui::End();
  return !open;
}





static void
glfw_error_callback(int error, const char* description)
{
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}



void
IMUI::run()
{
  glfwSetErrorCallback(glfw_error_callback);
  if(!glfwInit()) {
    fprintf(stderr, "Unable to init GLFW\n");
    exit(1);
  }
  GLFWwindow *window = glfwCreateWindow(1920, 1080, "Saga", NULL, NULL);
  if(window == NULL) {
    fprintf(stderr, "Unable to open window\n");
    exit(1);
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL2_Init();

  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  while(!glfwWindowShouldClose(window)) {
    handle();
    glfwPollEvents();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    auto windows = windows_;

    for(ssize_t i = windows.size() - 1; i >= 0; i--) {
      if(windows[i]->draw(*this)) {
        windows_.erase(windows_.begin() + i);
      }
    }


    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}


std::shared_ptr<UI>
createUI(void)
{
  return std::make_shared<IMUI>();

}

}

