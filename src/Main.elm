module Main exposing (main)

import Browser
import Html exposing (Html, div, text)
import Html.Attributes exposing (href)
import Material
import Material.Button as Button
import Material.Drawer.Persistent as PersistentDrawer
import Material.Drawer.Temporary as Drawer
import Material.Icon as Icon
import Material.List as List
import Material.Options as Options exposing (when)
import Material.Theme as Theme
import Material.TopAppBar as TopAppBar



-- MAIN


main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        }



-- MODEL


type alias Model =
    { mdc : Material.Model Msg
    , title : String
    , repo : String
    , isDrawerOpen : Bool
    }


defaultModel =
    { mdc = Material.defaultModel
    , title = "张凯的博客"
    , repo = "https://github.com/kaizhang91/blog"
    , isDrawerOpen = False
    }


type Msg
    = Mdc (Material.Msg Msg)
    | OpenDrawer
    | CloseDrawer



-- VIEW


view : Model -> Html Msg
view model =
    div []
        [ TopAppBar.view
            Mdc
            "appBar"
            model.mdc
            []
            [ TopAppBar.section
                [ TopAppBar.alignStart
                ]
                [ TopAppBar.navigationIcon
                    [ Icon.button
                    , Options.onClick OpenDrawer
                    ]
                    "menu"
                , TopAppBar.title [] [ text model.title ]
                ]
            , TopAppBar.section
                [ TopAppBar.alignEnd
                ]
                [ TopAppBar.actionItem
                    [ Icon.anchor
                    , Options.attribute (href model.repo)
                    ]
                    "code"
                , TopAppBar.actionItem [] "print"
                ]
            ]
        , Drawer.view
            Mdc
            "drawer"
            model.mdc
            [ Drawer.open |> when model.isDrawerOpen
            , Drawer.onClose CloseDrawer
            ]
            [ Drawer.header
                [ Theme.primaryBg
                , Theme.textPrimaryOnPrimary
                ]
                [ Drawer.headerContent []
                    [ text model.title
                    ]
                ]
            , List.group
                [ PersistentDrawer.content
                ]
                [ List.nav []
                    [ List.a
                        [ Options.attribute (href model.repo)
                        ]
                        [ List.graphicIcon [] "code"
                        , text "源码"
                        ]
                    ]
                ]
            ]
        ]


init : () -> ( Model, Cmd Msg )
init _ =
    ( defaultModel, Material.init Mdc )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Mdc msg_ ->
            Material.update Mdc msg_ model

        OpenDrawer ->
            ( { model | isDrawerOpen = True }, Cmd.none )

        CloseDrawer ->
            ( { model | isDrawerOpen = False }, Cmd.none )


subscriptions : Model -> Sub Msg
subscriptions model =
    Material.subscriptions Mdc model
