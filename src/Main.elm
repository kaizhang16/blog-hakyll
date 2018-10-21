module Main exposing (main)

import Browser
import Html exposing (Html, div, text)
import Material
import Material.Button as Button
import Material.Options as Options



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
    , content : String
    }


defaultModel =
    { mdc = Material.defaultModel
    , content = "unclicked"
    }


type Msg
    = Mdc (Material.Msg Msg)
    | Click
    | Reset



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Material.subscriptions Mdc model



-- VIEW


view : Model -> Html Msg
view model =
    div []
        [ Button.view Mdc
            "click"
            model.mdc
            [ Button.ripple
            , Options.onClick Click
            ]
            [ text "Click me!" ]
        , Button.view Mdc
            "reset"
            model.mdc
            [ Button.ripple
            , Options.onClick Reset
            ]
            [ text "Reset" ]
        , div [] [ text model.content ]
        ]


init : () -> ( Model, Cmd Msg )
init _ =
    ( defaultModel, Material.init Mdc )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Mdc msg_ ->
            Material.update Mdc msg_ model

        Click ->
            ( { model | content = "clicked" }
            , Cmd.none
            )

        Reset ->
            ( { model | content = "unclicked" }
            , Cmd.none
            )
